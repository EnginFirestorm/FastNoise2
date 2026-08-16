#include <cfloat>

#include "PorismNodes.h"
#include "Utils.inl"

// Bit-parity discipline for every kernel in this file: where a node replaces a
// chain of existing primitives, it must run the SAME FS ops in the SAME order
// and association as those primitives' kernels. In particular:
//  - a multiply followed by an add stays two ops (no FS::FMulAdd fusion),
//  - constants are splatted the way HybridSource constants are,
//  - Min/Max operand order matches the chain (minps/maxps are not commutative
//    for NaN and signed zero).

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::FractalBillow, SIMD> final : public virtual FastNoise::FractalBillow, public DispatchClass<FastNoise::Fractal<>, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        float32v gain = this->GetSourceValue( mGain, seed, pos... );
        float32v weightedStrength = this->GetSourceValue( mWeightedStrength, seed, pos... );
        float32v lacunarity( mLacunarity );
        float32v amp( 1.0f );
        float32v noise = FS::FMulAdd( FS::Abs( this->GetSourceValue( mSource, seed, pos... ) ), float32v( 2 ), float32v( -1 ) );

        float32v sum = noise * amp;

        for( int i = 1; i < mOctaves; i++ )
        {
            seed -= int32v( -1 );
            amp *= Lerp( float32v( 1 ), (noise + float32v( 1 )) * float32v( 0.5f ), weightedStrength );
            amp *= gain;

            noise = FS::FMulAdd( FS::Abs( this->GetSourceValue( mSource, seed, (pos *= lacunarity)... ) ), float32v( 2 ), float32v( -1 ) );
            sum += noise * amp;
        }

        return sum;
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::FractalFBmNormalized, SIMD> final : public virtual FastNoise::FractalFBmNormalized, public DispatchClass<FastNoise::Fractal<>, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        // FractalFBm's loop verbatim, plus a running amplitude sum. The final
        // divide is a true division: unlike the constant 1/sum multiply the
        // converter writes, this stays exact when Gain varies per sample.
        float32v gain = this->GetSourceValue( mGain, seed, pos... );
        float32v weightedStrength = this->GetSourceValue( mWeightedStrength, seed, pos... );
        float32v lacunarity( mLacunarity );
        float32v amp( 1.0f );
        float32v ampSum = amp;
        float32v noise = this->GetSourceValue( mSource, seed, pos... );

        float32v sum = noise * amp;

        for( int i = 1; i < mOctaves; i++ )
        {
            seed -= int32v( -1 );
            amp *= Lerp( float32v( 1 ), (noise + float32v( 1 )) * float32v( 0.5f ), weightedStrength );
            amp *= gain;
            ampSum += amp;

            noise = this->GetSourceValue( mSource, seed, (pos *= lacunarity)... );
            sum += noise * amp;
        }

        return sum / ampSum;
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::FractalHybridMulti, SIMD> final : public virtual FastNoise::FractalHybridMulti, public DispatchClass<FastNoise::Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        float32v lacunarity( mLacunarity );
        float32v offset( mOffset );
        float32v value = this->GetSourceValue( mSource, seed, pos... ) + offset;
        float32v weight = value;

        for( int i = 1; i < mOctaves; i++ )
        {
            weight = FS::Min( weight, float32v( 1 ) );
            seed -= int32v( -1 );

            float32v signal = ( this->GetSourceValue( mSource, seed, (pos *= lacunarity)... ) + offset )
                              * float32v( mSpectralWeights[i] );

            value = FS::FMulAdd( weight, signal, value );
            weight *= signal;
        }

        return value;
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::BiasGain, SIMD> final : public virtual FastNoise::BiasGain, public DispatchClass<FastNoise::Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        // Schlick bias then gain on t in [0,1], gain applied on the mirrored
        // half-interval. True divisions, not FS::Reciprocal: the approximate
        // reciprocal differs per SIMD level and this node has no primitive
        // chain to be bit-compared against - stability across levels wins.
        float32v t = FS::FMulAdd( this->GetSourceValue( mSource, seed, pos... ), float32v( 0.5f ), float32v( 0.5f ) );

        float32v kB( mBiasK );
        t = t / FS::FMulAdd( kB, float32v( 1 ) - t, float32v( 1 ) );

        mask32v hi = t > float32v( 0.5f );
        float32v u = FS::Select( hi, float32v( 1 ) - t, t );

        float32v kG( mGainK );
        u = u / FS::FMulAdd( kG, float32v( 1 ) - u - u, float32v( 1 ) );

        t = FS::Select( hi, float32v( 1 ) - u, u );
        return FS::FMulAdd( t, float32v( 2 ), float32v( -1 ) );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::Threshold, SIMD> final : public virtual FastNoise::Threshold, public DispatchClass<FastNoise::Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        float32v v = this->GetSourceValue( mSource, seed, pos... );
        float32v t = this->GetSourceValue( mThreshold, seed, pos... );
        float32v below = this->GetSourceValue( mBelow, seed, pos... );
        float32v above = this->GetSourceValue( mAbove, seed, pos... );

        return FS::Select( v < t, below, above );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::Clamp, SIMD> final : public virtual FastNoise::Clamp, public DispatchClass<FastNoise::Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        // Exactly Min(Max(source, min), max) - the chain this node replaces.
        return FS::Min( FS::Max( this->GetSourceValue( mSource, seed, pos... ),
                                 this->GetSourceValue( mMin, seed, pos... ) ),
                        this->GetSourceValue( mMax, seed, pos... ) );
    }
};

// The library builds with /fp:fast / -ffast-math, which LICENSES the compiler
// to contract mul+add into one FMA inside a function - the primitive chain
// cannot be contracted because its mul and add sit behind a virtual call
// boundary. Caught by FastNoise.BitEqual.ScaleBiasMatchesChain as a 1-ulp
// drift; contraction is therefore switched off for exactly this kernel.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma fp_contract( off )
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize( "fp-contract=off" )
#endif

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::ScaleBias, SIMD> final : public virtual FastNoise::ScaleBias, public DispatchClass<FastNoise::Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
#if defined(__clang__)
#pragma clang fp contract( off )
#endif
        // Two separate ops on purpose - see the bit-parity note at the top.
        float32v scaled = this->GetSourceValue( mSource, seed, pos... ) * this->GetSourceValue( mScale, seed, pos... );
        return scaled + this->GetSourceValue( mBias, seed, pos... );
    }
};

#if defined(_MSC_VER) && !defined(__clang__)
#pragma fp_contract( on )
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::Curve, SIMD> final : public virtual FastNoise::Curve, public DispatchClass<FastNoise::Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        float32v v = this->GetSourceValue( mSource, seed, pos... );

        // Branchless segment scan: each active segment overwrites the result
        // for lanes past its start. Clamped t makes the last segment extend
        // to +inf and the first Out covers everything below the first point.
        float32v result( mPointOut[0] );

        for( int i = 0; i < mPointCount - 1; i++ )
        {
            float32v inA( mPointIn[i] );
            float32v inB( mPointIn[i + 1] );

            float32v t = ( v - inA ) / ( inB - inA );
            t = FS::Max( float32v( 0 ), FS::Min( float32v( 1 ), t ) );

            float32v seg = Lerp( float32v( mPointOut[i] ), float32v( mPointOut[i + 1] ), t );
            result = FS::Select( v > inA, seg, result );
        }

        return result;
    }
};

// The two Simple cellular kernels below are verbatim copies of the full
// CellularValue / CellularDistance kernels (Cellular.inl) with the reduced
// configuration substituted where the full node branches:
//  - CalcDistance switch  -> inlined EuclideanSquared sequence (same op order),
//  - size-jitter branch   -> gone (identical to the full node at SizeJitter 0),
//  - slot bookkeeping     -> collapsed to the slot-0 operations, which are
//    bit-identical to what the full loops leave in slot 0,
//  - GetReturn            -> the Index0 / EuclideanSquared path only.
// The jitter constants must match Cellular.inl exactly.

namespace PorismCellularDetail
{
    static constexpr float kJitter2D = 0.437016f;
    static constexpr float kJitter3D = 0.396144f;
    static constexpr float kJitter4D = 0.366025f;
}

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::CellularValueSimple, SIMD> final : public virtual FastNoise::CellularValueSimple, public DispatchClass<FastNoise::VariableRange<FastNoise::Seeded<FastNoise::ScalableGenerator>>, SIMD>
{
    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y ) const
    {
        int32v sourceSeed = seed;
        seed += int32v( this->mSeedOffset );
        float32v jitter = float32v( PorismCellularDetail::kJitter2D ) * this->GetSourceValue( mGridJitter, sourceSeed, x, y );

        // Uninitialised on purpose, like the full node's valueHash array: the
        // first cell always wins against kInfinity and overwrites it.
        int32v valueHash0;
        float32v distance0( kInfinity );

        this->ScalePositions( x, y );

        int32v xc = FS::Convert<int32_t>( x ) + int32v( -1 );
        int32v ycBase = FS::Convert<int32_t>( y ) + int32v( -1 );

        float32v xcf = FS::Convert<float>( xc ) - x;
        float32v ycfBase = FS::Convert<float>( ycBase ) - y;

        xc *= int32v( Primes::X );
        ycBase *= int32v( Primes::Y );

        for( int xi = 0; xi < 3; xi++ )
        {
            float32v ycf = ycfBase;
            int32v yc = ycBase;
            for( int yi = 0; yi < 3; yi++ )
            {
                int32v hash = HashPrimesHB( seed, xc, yc );
                float32v xd = FS::Convert<float>( hash & int32v( 0x7ff ) ) - float32v( 0x7ff / 2.0f );
                float32v yd = FS::Convert<float>( FS::BitShiftRightZeroExtend( hash, 21 ) ) - float32v( 0x7ff / 2.0f );

                float32v invMag = jitter * FS::InvSqrt( FS::FMulAdd( xd, xd, yd * yd ) );
                xd = FS::FMulAdd( xd, invMag, xcf );
                yd = FS::FMulAdd( yd, invMag, ycf );

                float32v newDistance = xd * xd;
                newDistance = FS::FMulAdd( yd, yd, newDistance );

                mask32v closer = newDistance < distance0;
                distance0 = FS::Select( closer, newDistance, distance0 );
                valueHash0 = FS::Select( closer, hash, valueHash0 );

                ycf += float32v( 1 );
                yc += int32v( Primes::Y );
            }
            xcf += float32v( 1 );
            xc += int32v( Primes::X );
        }

        return this->ScaleOutput( FS::Convert<float>( valueHash0 ), -kValueBounds, kValueBounds );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z ) const
    {
        int32v sourceSeed = seed;
        seed += int32v( this->mSeedOffset );
        float32v jitter = float32v( PorismCellularDetail::kJitter3D ) * this->GetSourceValue( mGridJitter, sourceSeed, x, y, z );

        int32v valueHash0;
        float32v distance0( kInfinity );

        this->ScalePositions( x, y, z );

        int32v xc = FS::Convert<int32_t>( x ) + int32v( -1 );
        int32v ycBase = FS::Convert<int32_t>( y ) + int32v( -1 );
        int32v zcBase = FS::Convert<int32_t>( z ) + int32v( -1 );

        float32v xcf = FS::Convert<float>( xc ) - x;
        float32v ycfBase = FS::Convert<float>( ycBase ) - y;
        float32v zcfBase = FS::Convert<float>( zcBase ) - z;

        xc *= int32v( Primes::X );
        ycBase *= int32v( Primes::Y );
        zcBase *= int32v( Primes::Z );

        for( int xi = 0; xi < 3; xi++ )
        {
            float32v ycf = ycfBase;
            int32v yc = ycBase;
            for( int yi = 0; yi < 3; yi++ )
            {
                float32v zcf = zcfBase;
                int32v zc = zcBase;
                for( int zi = 0; zi < 3; zi++ )
                {
                    int32v hash = HashPrimesHB( seed, xc, yc, zc );
                    float32v xd = FS::Convert<float>( hash & int32v( 0x3ff ) ) - float32v( 0x3ff / 2.0f );
                    float32v yd = FS::Convert<float>( ( hash >> 11 ) & int32v( 0x3ff ) ) - float32v( 0x3ff / 2.0f );
                    float32v zd = FS::Convert<float>( FS::BitShiftRightZeroExtend( hash, 22 ) ) - float32v( 0x3ff / 2.0f );

                    float32v invMag = jitter * FS::InvSqrt( FS::FMulAdd( xd, xd, FS::FMulAdd( yd, yd, zd * zd ) ) );
                    xd = FS::FMulAdd( xd, invMag, xcf );
                    yd = FS::FMulAdd( yd, invMag, ycf );
                    zd = FS::FMulAdd( zd, invMag, zcf );

                    float32v newDistance = xd * xd;
                    newDistance = FS::FMulAdd( yd, yd, newDistance );
                    newDistance = FS::FMulAdd( zd, zd, newDistance );

                    mask32v closer = newDistance < distance0;
                    distance0 = FS::Select( closer, newDistance, distance0 );
                    valueHash0 = FS::Select( closer, hash, valueHash0 );

                    zcf += float32v( 1 );
                    zc += int32v( Primes::Z );
                }
                ycf += float32v( 1 );
                yc += int32v( Primes::Y );
            }
            xcf += float32v( 1 );
            xc += int32v( Primes::X );
        }

        return this->ScaleOutput( FS::Convert<float>( valueHash0 ), -kValueBounds, kValueBounds );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z, float32v w ) const
    {
        int32v sourceSeed = seed;
        seed += int32v( this->mSeedOffset );
        float32v jitter = float32v( PorismCellularDetail::kJitter4D ) * this->GetSourceValue( mGridJitter, sourceSeed, x, y, z, w );

        int32v valueHash0;
        float32v distance0( kInfinity );

        this->ScalePositions( x, y, z, w );

        int32v xc = FS::Convert<int32_t>( x ) + int32v( -1 );
        int32v ycBase = FS::Convert<int32_t>( y ) + int32v( -1 );
        int32v zcBase = FS::Convert<int32_t>( z ) + int32v( -1 );
        int32v wcBase = FS::Convert<int32_t>( w ) + int32v( -1 );

        float32v xcf = FS::Convert<float>( xc ) - x;
        float32v ycfBase = FS::Convert<float>( ycBase ) - y;
        float32v zcfBase = FS::Convert<float>( zcBase ) - z;
        float32v wcfBase = FS::Convert<float>( wcBase ) - w;

        xc *= int32v( Primes::X );
        ycBase *= int32v( Primes::Y );
        zcBase *= int32v( Primes::Z );
        wcBase *= int32v( Primes::W );

        for( int xi = 0; xi < 3; xi++ )
        {
            float32v ycf = ycfBase;
            int32v yc = ycBase;
            for( int yi = 0; yi < 3; yi++ )
            {
                float32v zcf = zcfBase;
                int32v zc = zcBase;
                for( int zi = 0; zi < 3; zi++ )
                {
                    float32v wcf = wcfBase;
                    int32v wc = wcBase;
                    for( int wi = 0; wi < 3; wi++ )
                    {
                        int32v hash = HashPrimesHB( seed, xc, yc, zc, wc );
                        float32v xd = FS::Convert<float>( hash & int32v( 0xff ) ) - float32v( 0xff / 2.0f );
                        float32v yd = FS::Convert<float>( (hash >> 8) & int32v( 0xff ) ) - float32v( 0xff / 2.0f );
                        float32v zd = FS::Convert<float>( (hash >> 16) & int32v( 0xff ) ) - float32v( 0xff / 2.0f );
                        float32v wd = FS::Convert<float>( FS::BitShiftRightZeroExtend( hash, 24 ) ) - float32v( 0xff / 2.0f );

                        float32v invMag = jitter * FS::InvSqrt( FS::FMulAdd( xd, xd, FS::FMulAdd( yd, yd, FS::FMulAdd( zd, zd, wd * wd ) ) ) );
                        xd = FS::FMulAdd( xd, invMag, xcf );
                        yd = FS::FMulAdd( yd, invMag, ycf );
                        zd = FS::FMulAdd( zd, invMag, zcf );
                        wd = FS::FMulAdd( wd, invMag, wcf );

                        float32v newDistance = xd * xd;
                        newDistance = FS::FMulAdd( yd, yd, newDistance );
                        newDistance = FS::FMulAdd( zd, zd, newDistance );
                        newDistance = FS::FMulAdd( wd, wd, newDistance );

                        mask32v closer = newDistance < distance0;
                        distance0 = FS::Select( closer, newDistance, distance0 );
                        valueHash0 = FS::Select( closer, hash, valueHash0 );

                        wcf += float32v( 1 );
                        wc += int32v( Primes::W );
                    }
                    zcf += float32v( 1 );
                    zc += int32v( Primes::Z );
                }
                ycf += float32v( 1 );
                yc += int32v( Primes::Y );
            }
            xcf += float32v( 1 );
            xc += int32v( Primes::X );
        }

        return this->ScaleOutput( FS::Convert<float>( valueHash0 ), -kValueBounds, kValueBounds );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::CellularDistanceSimple, SIMD> final : public virtual FastNoise::CellularDistanceSimple, public DispatchClass<FastNoise::VariableRange<FastNoise::Seeded<FastNoise::ScalableGenerator>>, SIMD>
{
    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y ) const
    {
        int32v sourceSeed = seed;
        seed += int32v( this->mSeedOffset );
        float32v jitter = float32v( PorismCellularDetail::kJitter2D ) * this->GetSourceValue( mGridJitter, sourceSeed, x, y );

        float32v distance0( kInfinity );

        this->ScalePositions( x, y );

        int32v xc = FS::Convert<int32_t>( x ) + int32v( -1 );
        int32v ycBase = FS::Convert<int32_t>( y ) + int32v( -1 );

        float32v xcf = FS::Convert<float>( xc );
        float32v ycfBase = FS::Convert<float>( ycBase );

        xc *= int32v( Primes::X );
        ycBase *= int32v( Primes::Y );

        for( int xi = 0; xi < 3; xi++ )
        {
            float32v xcfOffset = xcf - x;
            float32v ycf = ycfBase;
            int32v yc = ycBase;
            for( int yi = 0; yi < 3; yi++ )
            {
                int32v hash = HashPrimesHB( seed, xc, yc );
                float32v xd = FS::Convert<float>( hash & int32v( 0x7ff ) ) - float32v( 0x7ff / 2.0f );
                float32v yd = FS::Convert<float>( FS::BitShiftRightZeroExtend( hash, 21 ) ) - float32v( 0x7ff / 2.0f );

                float32v invMag = jitter * FS::InvSqrt( FS::FMulAdd( xd, xd, yd * yd ) );
                xd = FS::FMulAdd( xd, invMag, xcfOffset );
                yd = FS::FMulAdd( yd, invMag, ycf - y );

                float32v newDistance = xd * xd;
                newDistance = FS::FMulAdd( yd, yd, newDistance );

                distance0 = FS::Min( distance0, newDistance );

                ycf += float32v( 1 );
                yc += int32v( Primes::Y );
            }
            xcf += float32v( 1 );
            xc += int32v( Primes::X );
        }

        float maxDist = 1 + PorismCellularDetail::kJitter2D;
        maxDist *= maxDist;
        return this->ScaleOutput( distance0, 0, maxDist );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z ) const
    {
        int32v sourceSeed = seed;
        seed += int32v( this->mSeedOffset );
        float32v jitter = float32v( PorismCellularDetail::kJitter3D ) * this->GetSourceValue( mGridJitter, sourceSeed, x, y, z );

        float32v distance0( kInfinity );

        this->ScalePositions( x, y, z );

        int32v xc = FS::Convert<int32_t>( x ) + int32v( -1 );
        int32v ycBase = FS::Convert<int32_t>( y ) + int32v( -1 );
        int32v zcBase = FS::Convert<int32_t>( z ) + int32v( -1 );

        float32v xcf = FS::Convert<float>( xc );
        float32v ycfBase = FS::Convert<float>( ycBase );
        float32v zcfBase = FS::Convert<float>( zcBase );

        xc *= int32v( Primes::X );
        ycBase *= int32v( Primes::Y );
        zcBase *= int32v( Primes::Z );

        for( int xi = 0; xi < 3; xi++ )
        {
            float32v xcfOffset = xcf - x;
            float32v ycf = ycfBase;
            int32v yc = ycBase;
            for( int yi = 0; yi < 3; yi++ )
            {
                float32v ycfOffset = ycf - y;
                float32v zcf = zcfBase;
                int32v zc = zcBase;
                for( int zi = 0; zi < 3; zi++ )
                {
                    int32v hash = HashPrimesHB( seed, xc, yc, zc );
                    float32v xd = FS::Convert<float>( hash & int32v( 0x3ff ) ) - float32v( 0x3ff / 2.0f );
                    float32v yd = FS::Convert<float>( (hash >> 11) & int32v( 0x3ff ) ) - float32v( 0x3ff / 2.0f );
                    float32v zd = FS::Convert<float>( FS::BitShiftRightZeroExtend( hash, 22 ) ) - float32v( 0x3ff / 2.0f );

                    float32v invMag = jitter * FS::InvSqrt( FS::FMulAdd( xd, xd, FS::FMulAdd( yd, yd, zd * zd ) ) );
                    xd = FS::FMulAdd( xd, invMag, xcfOffset );
                    yd = FS::FMulAdd( yd, invMag, ycfOffset );
                    zd = FS::FMulAdd( zd, invMag, zcf - z );

                    float32v newDistance = xd * xd;
                    newDistance = FS::FMulAdd( yd, yd, newDistance );
                    newDistance = FS::FMulAdd( zd, zd, newDistance );

                    distance0 = FS::Min( distance0, newDistance );

                    zcf += float32v( 1 );
                    zc += int32v( Primes::Z );
                }
                ycf += float32v( 1 );
                yc += int32v( Primes::Y );
            }
            xcf += float32v( 1 );
            xc += int32v( Primes::X );
        }

        float maxDist = 1 + PorismCellularDetail::kJitter3D;
        maxDist *= maxDist;
        return this->ScaleOutput( distance0, 0, maxDist );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z, float32v w ) const
    {
        int32v sourceSeed = seed;
        seed += int32v( this->mSeedOffset );
        float32v jitter = float32v( PorismCellularDetail::kJitter4D ) * this->GetSourceValue( mGridJitter, sourceSeed, x, y, z, w );

        float32v distance0( kInfinity );

        this->ScalePositions( x, y, z, w );

        int32v xc = FS::Convert<int32_t>( x ) + int32v( -1 );
        int32v ycBase = FS::Convert<int32_t>( y ) + int32v( -1 );
        int32v zcBase = FS::Convert<int32_t>( z ) + int32v( -1 );
        int32v wcBase = FS::Convert<int32_t>( w ) + int32v( -1 );

        float32v xcf = FS::Convert<float>( xc );
        float32v ycfBase = FS::Convert<float>( ycBase );
        float32v zcfBase = FS::Convert<float>( zcBase );
        float32v wcfBase = FS::Convert<float>( wcBase );

        xc *= int32v( Primes::X );
        ycBase *= int32v( Primes::Y );
        zcBase *= int32v( Primes::Z );
        wcBase *= int32v( Primes::W );

        for( int xi = 0; xi < 3; xi++ )
        {
            float32v xcfOffset = xcf - x;
            float32v ycf = ycfBase;
            int32v yc = ycBase;
            for( int yi = 0; yi < 3; yi++ )
            {
                float32v ycfOffset = ycf - y;
                float32v zcf = zcfBase;
                int32v zc = zcBase;
                for( int zi = 0; zi < 3; zi++ )
                {
                    float32v zcfOffset = zcf - z;
                    float32v wcf = wcfBase;
                    int32v wc = wcBase;
                    for( int wi = 0; wi < 3; wi++ )
                    {
                        int32v hash = HashPrimesHB( seed, xc, yc, zc, wc );
                        float32v xd = FS::Convert<float>( hash & int32v( 0xff ) ) - float32v( 0xff / 2.0f );
                        float32v yd = FS::Convert<float>( (hash >> 8) & int32v( 0xff ) ) - float32v( 0xff / 2.0f );
                        float32v zd = FS::Convert<float>( (hash >> 16) & int32v( 0xff ) ) - float32v( 0xff / 2.0f );
                        float32v wd = FS::Convert<float>( FS::BitShiftRightZeroExtend( hash, 24 ) ) - float32v( 0xff / 2.0f );

                        float32v invMag = jitter * FS::InvSqrt( FS::FMulAdd( xd, xd, FS::FMulAdd( yd, yd, FS::FMulAdd( zd, zd, wd * wd ) ) ) );
                        xd = FS::FMulAdd( xd, invMag, xcfOffset );
                        yd = FS::FMulAdd( yd, invMag, ycfOffset );
                        zd = FS::FMulAdd( zd, invMag, zcfOffset );
                        wd = FS::FMulAdd( wd, invMag, wcf - w );

                        float32v newDistance = xd * xd;
                        newDistance = FS::FMulAdd( yd, yd, newDistance );
                        newDistance = FS::FMulAdd( zd, zd, newDistance );
                        newDistance = FS::FMulAdd( wd, wd, newDistance );

                        distance0 = FS::Min( distance0, newDistance );

                        wcf += float32v( 1 );
                        wc += int32v( Primes::W );
                    }
                    zcf += float32v( 1 );
                    zc += int32v( Primes::Z );
                }
                ycf += float32v( 1 );
                yc += int32v( Primes::Y );
            }
            xcf += float32v( 1 );
            xc += int32v( Primes::X );
        }

        float maxDist = 1 + PorismCellularDetail::kJitter4D;
        maxDist *= maxDist;
        return this->ScaleOutput( distance0, 0, maxDist );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::BoxDomain, SIMD> final : public virtual FastNoise::BoxDomain, public DispatchClass<FastNoise::Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    // Strict: full Gradient fold over every dimension, dead-axis terms
    // included. `(pos_d + 0) * 0` is not a bit no-op - it produces signed
    // zeros that feed minps - so reproducing the chain means reproducing
    // these terms in dimension order with accumulator start +0.
    template<typename... P>
    FS_FORCEINLINE float32v PlaneStrict( size_t axis, const FastNoise::HybridSource& offset, float multiplier, int32v seed, P... pos ) const
    {
        size_t d = 0;
        float32v r( 0 );
        ( ( r = FS::FMulAdd( pos + ( d == axis ? this->GetSourceValue( offset, seed, pos... ) : float32v( 0 ) ),
                             float32v( d == axis ? multiplier : 0.0f ), r ), ++d ), ... );
        return r;
    }

    // Fast: only the live axis. Cheaper, NOT bit-identical to the chain.
    template<typename... P>
    FS_FORCEINLINE float32v PlaneFast( size_t axis, const FastNoise::HybridSource& offset, float multiplier, int32v seed, P... pos ) const
    {
        float32v posArr[] = { pos... };
        return ( posArr[axis] + this->GetSourceValue( offset, seed, pos... ) ) * float32v( multiplier );
    }

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        constexpr size_t kDims = sizeof...( P );

        // Hard Min iff smoothness is the constant 0 with no node attached -
        // the MinSmooth maths clamps smoothness to FLT_MIN and is therefore
        // not bit-identical to Min even as smoothness approaches zero.
        const bool hardMin = mSmoothness.simdGeneratorPtr == nullptr && mSmoothness.constant == 0.0f;
        float32v smoothness( 0 );
        if( !hardMin )
        {
            smoothness = FS::Max( float32v( FLT_MIN ), FS::Abs( this->GetSourceValue( mSmoothness, seed, pos... ) ) );
        }

        bool haveAny = false;
        float32v acc( 0 );

        for( size_t axis = 0; axis < kDims; axis++ )
        {
            for( int bank = 0; bank < 2; bank++ )
            {
                const float multiplier = bank == 0 ? mLoMultiplier[axis] : mHiMultiplier[axis];
                if( multiplier == 0.0f )
                {
                    continue; // plane off
                }
                const FastNoise::HybridSource& offset = bank == 0 ? mLoOffset[axis] : mHiOffset[axis];

                float32v plane = mMode == FastNoise::BoxDomainMode::Strict
                    ? PlaneStrict( axis, offset, multiplier, seed, pos... )
                    : PlaneFast( axis, offset, multiplier, seed, pos... );

                if( !haveAny )
                {
                    acc = plane;
                    haveAny = true;
                }
                else if( hardMin )
                {
                    acc = FS::Min( acc, plane );
                }
                else
                {
                    // MinSmooth's exact sequence (Blends.inl), folded pairwise
                    // in the documented plane order.
                    float32v h = FS::Max( smoothness - FS::Abs( acc - plane ), float32v( 0.0f ) );
                    h *= FS::Reciprocal( smoothness );
                    acc = FS::FNMulAdd( float32v( 1.0f / 6.0f ), h * h * h * smoothness, FS::Min( acc, plane ) );
                }
            }
        }

        return haveAny ? acc : float32v( 0 );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::Slope, SIMD> final : public virtual FastNoise::Slope, public DispatchClass<FastNoise::Generator, SIMD>
{
    // One-sided differences: 1 + dimensions source evaluations per sample.
    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y ) const
    {
        float32v h( mStepSize );
        float32v invH( 1.0f / mStepSize );
        float32v n = this->GetSourceValue( mSource, seed, x, y );
        float32v dx = ( this->GetSourceValue( mSource, seed, x + h, y ) - n ) * invH;
        float32v dy = ( this->GetSourceValue( mSource, seed, x, y + h ) - n ) * invH;
        return FS::Sqrt( FS::FMulAdd( dx, dx, dy * dy ) );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z ) const
    {
        float32v h( mStepSize );
        float32v invH( 1.0f / mStepSize );
        float32v n = this->GetSourceValue( mSource, seed, x, y, z );
        float32v dx = ( this->GetSourceValue( mSource, seed, x + h, y, z ) - n ) * invH;
        float32v dy = ( this->GetSourceValue( mSource, seed, x, y + h, z ) - n ) * invH;
        float32v dz = ( this->GetSourceValue( mSource, seed, x, y, z + h ) - n ) * invH;
        return FS::Sqrt( FS::FMulAdd( dx, dx, FS::FMulAdd( dy, dy, dz * dz ) ) );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z, float32v w ) const
    {
        float32v h( mStepSize );
        float32v invH( 1.0f / mStepSize );
        float32v n = this->GetSourceValue( mSource, seed, x, y, z, w );
        float32v dx = ( this->GetSourceValue( mSource, seed, x + h, y, z, w ) - n ) * invH;
        float32v dy = ( this->GetSourceValue( mSource, seed, x, y + h, z, w ) - n ) * invH;
        float32v dz = ( this->GetSourceValue( mSource, seed, x, y, z + h, w ) - n ) * invH;
        float32v dw = ( this->GetSourceValue( mSource, seed, x, y, z, w + h ) - n ) * invH;
        return FS::Sqrt( FS::FMulAdd( dx, dx, FS::FMulAdd( dy, dy, FS::FMulAdd( dz, dz, dw * dw ) ) ) );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::Spheres, SIMD> final : public virtual FastNoise::Spheres, public DispatchClass<FastNoise::VariableRange<FastNoise::ScalableGenerator>, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        this->ScalePositions( pos... );

        float32v distSqr( 0 );
        ( ( distSqr = FS::FMulAdd( pos, pos, distSqr ) ), ... );
        float32v dist = FS::Sqrt( distSqr );

        float32v distFrac = dist - FS::Floor( dist );
        float32v nearest = FS::Min( distFrac, float32v( 1 ) - distFrac );

        return this->ScaleOutput( float32v( 1 ) - nearest * float32v( 4 ), -1, 1 );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::Cylinders, SIMD> final : public virtual FastNoise::Cylinders, public DispatchClass<FastNoise::VariableRange<FastNoise::ScalableGenerator>, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        this->ScalePositions( pos... );

        const float32v posArr[] = { pos... };
        float32v distSqr = posArr[0] * posArr[0];
        if constexpr( sizeof...( P ) > 1 )
        {
            distSqr = FS::FMulAdd( posArr[1], posArr[1], distSqr );
        }
        float32v dist = FS::Sqrt( distSqr );

        float32v distFrac = dist - FS::Floor( dist );
        float32v nearest = FS::Min( distFrac, float32v( 1 ) - distFrac );

        return this->ScaleOutput( float32v( 1 ) - nearest * float32v( 4 ), -1, 1 );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FastNoise::FractalErosion, SIMD> final : public virtual FastNoise::FractalErosion, public DispatchClass<FastNoise::Generator, SIMD>
{
    // iq / deCarpentier swiss-turbulence shape with finite-difference
    // gradients: later octaves are damped where the accumulated gradient is
    // steep. 1 + dimensions source evaluations per octave.
    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y ) const
    {
        float32v gain = this->GetSourceValue( mGain, seed, x, y );
        float32v lacunarity( mLacunarity );
        float32v h( mStepSize );
        float32v invH( 1.0f / mStepSize );
        float32v erosion( mErosionStrength );
        float32v amp( 1.0f );
        float32v sum( 0.0f );
        float32v dsx( 0.0f ), dsy( 0.0f );

        for( int i = 0; i < mOctaves; i++ )
        {
            float32v n = this->GetSourceValue( mSource, seed, x, y );
            float32v dx = ( this->GetSourceValue( mSource, seed, x + h, y ) - n ) * invH;
            float32v dy = ( this->GetSourceValue( mSource, seed, x, y + h ) - n ) * invH;

            dsx += dx * amp;
            dsy += dy * amp;

            float32v damp = float32v( 1 ) + erosion * FS::FMulAdd( dsx, dsx, dsy * dsy );
            sum += amp * n / damp;

            seed -= int32v( -1 );
            x *= lacunarity;
            y *= lacunarity;
            amp *= gain;
        }
        return sum;
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z ) const
    {
        float32v gain = this->GetSourceValue( mGain, seed, x, y, z );
        float32v lacunarity( mLacunarity );
        float32v h( mStepSize );
        float32v invH( 1.0f / mStepSize );
        float32v erosion( mErosionStrength );
        float32v amp( 1.0f );
        float32v sum( 0.0f );
        float32v dsx( 0.0f ), dsy( 0.0f ), dsz( 0.0f );

        for( int i = 0; i < mOctaves; i++ )
        {
            float32v n = this->GetSourceValue( mSource, seed, x, y, z );
            float32v dx = ( this->GetSourceValue( mSource, seed, x + h, y, z ) - n ) * invH;
            float32v dy = ( this->GetSourceValue( mSource, seed, x, y + h, z ) - n ) * invH;
            float32v dz = ( this->GetSourceValue( mSource, seed, x, y, z + h ) - n ) * invH;

            dsx += dx * amp;
            dsy += dy * amp;
            dsz += dz * amp;

            float32v damp = float32v( 1 ) + erosion * FS::FMulAdd( dsx, dsx, FS::FMulAdd( dsy, dsy, dsz * dsz ) );
            sum += amp * n / damp;

            seed -= int32v( -1 );
            x *= lacunarity;
            y *= lacunarity;
            z *= lacunarity;
            amp *= gain;
        }
        return sum;
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z, float32v w ) const
    {
        float32v gain = this->GetSourceValue( mGain, seed, x, y, z, w );
        float32v lacunarity( mLacunarity );
        float32v h( mStepSize );
        float32v invH( 1.0f / mStepSize );
        float32v erosion( mErosionStrength );
        float32v amp( 1.0f );
        float32v sum( 0.0f );
        float32v dsx( 0.0f ), dsy( 0.0f ), dsz( 0.0f ), dsw( 0.0f );

        for( int i = 0; i < mOctaves; i++ )
        {
            float32v n = this->GetSourceValue( mSource, seed, x, y, z, w );
            float32v dx = ( this->GetSourceValue( mSource, seed, x + h, y, z, w ) - n ) * invH;
            float32v dy = ( this->GetSourceValue( mSource, seed, x, y + h, z, w ) - n ) * invH;
            float32v dz = ( this->GetSourceValue( mSource, seed, x, y, z + h, w ) - n ) * invH;
            float32v dw = ( this->GetSourceValue( mSource, seed, x, y, z, w + h ) - n ) * invH;

            dsx += dx * amp;
            dsy += dy * amp;
            dsz += dz * amp;
            dsw += dw * amp;

            float32v damp = float32v( 1 ) + erosion * FS::FMulAdd( dsx, dsx, FS::FMulAdd( dsy, dsy, FS::FMulAdd( dsz, dsz, dsw * dsw ) ) );
            sum += amp * n / damp;

            seed -= int32v( -1 );
            x *= lacunarity;
            y *= lacunarity;
            z *= lacunarity;
            w *= lacunarity;
            amp *= gain;
        }
        return sum;
    }
};
