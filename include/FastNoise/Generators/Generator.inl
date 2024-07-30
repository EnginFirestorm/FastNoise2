#include <cassert>
#include <cstring>
#include "FastSIMD/InlInclude.h"

#include "Generator.h"

#ifdef FS_SIMD_CLASS
#pragma warning( disable:4250 )
#endif

template<typename FS>
class FS_T<FastNoise::Generator, FS> : public virtual FastNoise::Generator
{
    FASTSIMD_DECLARE_FS_TYPES;

public:
    virtual float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y ) const = 0;
    virtual float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z ) const = 0;
    virtual float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z, float32v w ) const { return Gen( seed, x, y, z ); };

#define FASTNOISE_IMPL_GEN_T\
    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y ) const override { return GenT( seed, x, y ); }\
    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z ) const override { return GenT( seed, x, y, z ); }\
    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z, float32v w ) const override { return GenT( seed, x, y, z, w ); }

    FastSIMD::eLevel GetSIMDLevel() const final
    {
        return FS::SIMD_Level;
    }

    using VoidPtrStorageType = const FS_T<Generator, FS>*;

    void SetSourceSIMDPtr( const Generator* base, const void** simdPtr ) final
    {
        if( !base )
        {
            *simdPtr = nullptr;
            return;
        }
        auto simd = dynamic_cast<VoidPtrStorageType>( base );

        assert( simd );
        *simdPtr = reinterpret_cast<const void*>( simd );
    }

    template<typename T, typename... POS>
    FS_INLINE float32v FS_VECTORCALL GetSourceValue( const FastNoise::HybridSourceT<T>& memberVariable, int32v seed, POS... pos ) const
    {
        if( memberVariable.simdGeneratorPtr )
        {
            auto simdGen = reinterpret_cast<VoidPtrStorageType>( memberVariable.simdGeneratorPtr );

            return simdGen->Gen( seed, pos... );
        }
        return float32v( memberVariable.constant );
    }

    template<typename T, typename... POS>
    FS_INLINE float32v FS_VECTORCALL GetSourceValue( const FastNoise::GeneratorSourceT<T>& memberVariable, int32v seed, POS... pos ) const
    {
        assert( memberVariable.simdGeneratorPtr );
        auto simdGen = reinterpret_cast<VoidPtrStorageType>( memberVariable.simdGeneratorPtr );

        return simdGen->Gen( seed, pos... );
    }

    template<typename T>
    FS_INLINE const FS_T<T, FS>* GetSourceSIMD( const FastNoise::GeneratorSourceT<T>& memberVariable ) const
    {
        assert( memberVariable.simdGeneratorPtr );
        auto simdGen = reinterpret_cast<VoidPtrStorageType>( memberVariable.simdGeneratorPtr );

        auto simdT = static_cast<const FS_T<T, FS>*>( simdGen );
        return simdT;
    }

    FastNoise::OutputMinMax GenUniformGrid2D( float* noiseOut, int xStart, int yStart, int xSize, int ySize, float frequency, int seed ) const final
    {
        float32v min( INFINITY );
        float32v max( -INFINITY );

        int32v xIdx( xStart );
        int32v yIdx( yStart );

        float32v freqV( frequency );

        int32v xSizeV( xSize );
        int32v xMax = xSizeV + xIdx + int32v( -1 );

        intptr_t totalValues = xSize * ySize;
        intptr_t index = 0;

        xIdx += int32v::FS_Incremented();

        AxisReset<true>( xIdx, yIdx, xMax, xSizeV, xSize );

        while( index < totalValues - (intptr_t)FS_Size_32() )
        {
            float32v xPos = FS_Converti32_f32( xIdx ) * freqV;
            float32v yPos = FS_Converti32_f32( yIdx ) * freqV;

            float32v gen = Gen( int32v( seed ), xPos, yPos );
            FS_Store_f32( &noiseOut[index], gen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS_Min_f32( min, gen );
            max = FS_Max_f32( max, gen );
#endif

            index += FS_Size_32();
            xIdx += int32v( FS_Size_32() );

            AxisReset<false>( xIdx, yIdx, xMax, xSizeV, xSize );
        }

        float32v xPos = FS_Converti32_f32( xIdx ) * freqV;
        float32v yPos = FS_Converti32_f32( yIdx ) * freqV;

        float32v gen = Gen( int32v( seed ), xPos, yPos );

        return DoRemaining( noiseOut, totalValues, index, min, max, gen );
    }

    FastNoise::OutputMinMax GenUniformGrid3D( float* noiseOut, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float frequency, int seed ) const final
    {
        float32v min( INFINITY );
        float32v max( -INFINITY );

        int32v xIdx( xStart );
        int32v yIdx( yStart );
        int32v zIdx( zStart );

        float32v freqV( frequency );

        int32v xSizeV( xSize );
        int32v xMax = xSizeV + xIdx + int32v( -1 );
        int32v ySizeV( ySize );
        int32v yMax = ySizeV + yIdx + int32v( -1 );

        intptr_t totalValues = xSize * ySize * zSize;
        intptr_t index = 0;

        xIdx += int32v::FS_Incremented();

        AxisReset<true>( xIdx, yIdx, xMax, xSizeV, xSize );
        AxisReset<true>( yIdx, zIdx, yMax, ySizeV, xSize * ySize );

        while( index < totalValues - (intptr_t)FS_Size_32() )
        {
            float32v xPos = FS_Converti32_f32( xIdx ) * freqV;
            float32v yPos = FS_Converti32_f32( yIdx ) * freqV;
            float32v zPos = FS_Converti32_f32( zIdx ) * freqV;

            float32v gen = Gen( int32v( seed ), xPos, yPos, zPos );
            FS_Store_f32( &noiseOut[index], gen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS_Min_f32( min, gen );
            max = FS_Max_f32( max, gen );
#endif

            index += FS_Size_32();
            xIdx += int32v( FS_Size_32() );
            
            AxisReset<false>( xIdx, yIdx, xMax, xSizeV, xSize );
            AxisReset<false>( yIdx, zIdx, yMax, ySizeV, xSize * ySize );
        }

        float32v xPos = FS_Converti32_f32( xIdx ) * freqV;
        float32v yPos = FS_Converti32_f32( yIdx ) * freqV;
        float32v zPos = FS_Converti32_f32( zIdx ) * freqV;

        float32v gen = Gen( int32v( seed ), xPos, yPos, zPos );

        return DoRemaining( noiseOut, totalValues, index, min, max, gen );
    }

    void GenAxes( float* outX, float* outY, float* outZ, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, Generator* genX, Generator* genY, Generator* genZ, float frequencyX, float frequencyY, float frequencyZ, int seed ) const final
    {
        FS_T<FastNoise::Generator, FS>* genXFS = nullptr;
        if( genX )
            genXFS = dynamic_cast<FS_T<FastNoise::Generator, FS>*>( genX );
        FS_T<FastNoise::Generator, FS>* genYFS = nullptr;
        if( genY )
			genYFS = dynamic_cast<FS_T<FastNoise::Generator, FS>*>( genY );
        FS_T<FastNoise::Generator, FS>* genZFS = nullptr;
        if( genZ )
			genZFS = dynamic_cast<FS_T<FastNoise::Generator, FS>*>( genZ );

        int32v seedV( seed );

        int32v xIdx( xStart );
        int32v yIdx( yStart );
        int32v zIdx( zStart );

        float32v freqVX( frequencyX );
        float32v freqVY( frequencyY );
        float32v freqVZ( frequencyZ );

        int32v xSizeV( xSize );
        int32v xMax = xSizeV + xIdx + int32v( -1 );
        int32v ySizeV( ySize );
        int32v yMax = ySizeV + yIdx + int32v( -1 );

        intptr_t totalValues = xSize * ySize * zSize;
        intptr_t index = 0;

        xIdx += int32v::FS_Incremented();

        AxisReset<true>( xIdx, yIdx, xMax, xSizeV, xSize );
        AxisReset<true>( yIdx, zIdx, yMax, ySizeV, xSize * ySize );

        while( index < totalValues )
        {
            float32v xPos = FS_Converti32_f32( xIdx ) * freqVX;
            float32v yPos = FS_Converti32_f32( yIdx ) * freqVY;
            float32v zPos = FS_Converti32_f32( zIdx ) * freqVZ;

            float32v xOff;
            float32v yOff;
            float32v zOff;

            if( genXFS )
                xOff = genXFS->Gen( seedV, xPos, yPos, zPos );
            else
                xOff = float32v( 0.f );
            if( genYFS )
                yOff = genYFS->Gen( seedV, xPos, yPos, zPos );
            else
                yOff = float32v( 0.f );
            if( genZFS )
                zOff = genZFS->Gen( seedV, xPos, yPos, zPos );
            else
                zOff = float32v( 0.f );

            FS_Store_f32( &outX[index], xPos + xOff );
            FS_Store_f32( &outY[index], yPos + yOff );
            FS_Store_f32( &outZ[index], zPos + zOff );

            index += FS_Size_32();
            xIdx += int32v( FS_Size_32() );

            AxisReset<false>( xIdx, yIdx, xMax, xSizeV, xSize );
            AxisReset<false>( yIdx, zIdx, yMax, ySizeV, xSize * ySize );
        }
    }

    bool Gen3DDomainCheckSimpel( float* genX, float* genY, float* genZ, int seed, int size ) const final
    {
        int32v seedV( seed );

        intptr_t totalValues = size;
        intptr_t index = 0;

        float32v min( 0.f );

        while( index < totalValues )
        {
            float32v xPos = FS_Load_f32( &genX[index] );
            float32v yPos = FS_Load_f32( &genY[index] );
            float32v zPos = FS_Load_f32( &genZ[index] );

            if( FS_AnyMask_bool( Gen( seedV, xPos, yPos, zPos ) > min ) )
                return true;

            index += FS_Size_32();
        }
        return false;
    }

    bool Gen3DDomainCheck( float* genX, float* genY, float* genZ, int seed, int size, float biomover, float* biomPower ) const final
    {
        int32v seedV( seed );

        intptr_t totalValues = size;
        intptr_t index = 0;

        float32v min( 0.f );
        float32v over( biomover );

        while( index < totalValues )
        {
            float32v xPos = FS_Load_f32( &genX[index] );
            float32v yPos = FS_Load_f32( &genY[index] );
            float32v zPos = FS_Load_f32( &genZ[index] );
            float32v biomPowerV = FS_Load_f32( &biomPower[index] );

            float32v domain = Gen( seedV, xPos, yPos, zPos );

            if( FS_AnyMask_bool( ( domain + over ) > biomPowerV ) )
                return true;

            index += FS_Size_32();
        }
        return false;
    }

    void Gen3DComplexAdd( float* out, float* genX, float* genY, float* genZ, int seed, int size, float power ) const final
    {
        int32v seedV( seed );
        float32v powerV( power );

        intptr_t totalValues = size;
        intptr_t index = 0;

        while( index < totalValues )
        {
            float32v xPos = FS_Load_f32( &genX[index] );
            float32v yPos = FS_Load_f32( &genY[index] );
            float32v zPos = FS_Load_f32( &genZ[index] );

            FS_Store_f32( &out[index], FS_Load_f32( &out[index] ) + ( Gen( seedV, xPos, yPos, zPos ) * powerV ) );

            index += FS_Size_32();
        }
    }

        void Gen3DCompAddWV( float* out, float* domain, float* genX, float* genY, float* genZ, int seed, int size, float power ) const final
    {
        int32v seedV( seed );
        float32v powerV( power );

        intptr_t totalValues = size;
        intptr_t index = 0;
        float32v min( 0.f );
        float32v p( 1.f );

        while( index < totalValues )
        {
            float32v xPos = FS_Load_f32( &genX[index] );
            float32v yPos = FS_Load_f32( &genY[index] );
            float32v zPos = FS_Load_f32( &genZ[index] );

            mask32v mask = FS_Load_f32( &domain[index] ) <= min;
            float32v gen = FS_Load_f32( &out[index] ) + ( Gen( seedV, xPos, yPos, zPos ) * powerV );
            FS_Store_f32( &out[index], FS_Select_f32( mask, p, gen ) );

            index += FS_Size_32();
        }
    }

        void Gen3DAdd( float* noiseOut, float* genX, float* genY, float* genZ, float* genXoff, float* genYoff, float* genZoff, int seed, int size, Generator* genDomain) const final
    {
        FS_T<FastNoise::Generator, FS>* genDomainFS = dynamic_cast<FS_T<FastNoise::Generator, FS>*>( genDomain );

        int32v seedV( seed );

        intptr_t totalValues = size;
        intptr_t index = 0;

        float32v min( 0.f );
        float32v max( 1.f );

        while( index < totalValues )
        {
            float32v xPosOff = FS_Load_f32( &genXoff[index] );
            float32v yPosOff = FS_Load_f32( &genYoff[index] );
            float32v zPosOff = FS_Load_f32( &genZoff[index] );

            float32v domain = FS_Min_f32(FS_Max_f32( genDomainFS->Gen( seedV, xPosOff, yPosOff, zPosOff ), min ), max);

            if( FS_AnyMask_bool( domain > min ) )
            {
                float32v xPos = FS_Load_f32( &genX[index] );
                float32v yPos = FS_Load_f32( &genY[index] );
                float32v zPos = FS_Load_f32( &genZ[index] );

                FS_Store_f32( &noiseOut[index], ( FS_Load_f32( &noiseOut[index] ) + (Gen( seedV, xPos, yPos, zPos ) * domain) ) );
            }

            index += FS_Size_32();
        }
    }

    void Gen3DFullAdd( float* noiseOut, float* genX, float* genY, float* genZ,
                       float* genXoff, float* genYoff, float* genZoff,
                       int seed, int size, float overlap, float biomover,
                       Generator* genB, Generator* genPower, Generator* genDomain,
                       int biomIndex, float* biomPower, int* biomSwitch, int* biomList ) const final
    {
        FS_T<FastNoise::Generator, FS>* genBFS = dynamic_cast<FS_T<FastNoise::Generator, FS>*>( genB );
        FS_T<FastNoise::Generator, FS>* genPowerFS = dynamic_cast<FS_T<FastNoise::Generator, FS>*>( genPower );
        FS_T<FastNoise::Generator, FS>* genDomainFS = dynamic_cast<FS_T<FastNoise::Generator, FS>*>( genDomain );

        int32v seedV( seed );
        float32v overlapV( overlap );

        intptr_t totalValues = size;
        intptr_t index = 0;

        float32v min( 0.f );
        float32v over( biomover );

        int32v p( 1 );
        int32v m( -1 );
        int32v biomIndexV( biomIndex );

        while( index < totalValues )
        {
            float32v xPosOff = FS_Load_f32( &genXoff[index] );
            float32v yPosOff = FS_Load_f32( &genYoff[index] );
            float32v zPosOff = FS_Load_f32( &genZoff[index] );
            float32v biomPowerV = FS_Load_f32( &biomPower[index] );

            float32v domain = genDomainFS->Gen( seedV, xPosOff, yPosOff, zPosOff );

            if( FS_AnyMask_bool( ( domain + over ) > biomPowerV ) )
            {
                float32v xPos = FS_Load_f32( &genX[index] );
                float32v yPos = FS_Load_f32( &genY[index] );
                float32v zPos = FS_Load_f32( &genZ[index] );

                float32v basePow = genPowerFS->Gen( seedV, xPos, yPos, zPos );
                float32v powA = FS_Max_f32( overlapV + basePow, min );
                float32v powB = FS_Max_f32( overlapV - basePow, min );

                float32v out = float32v(0.f );

                if( FS_AnyMask_bool( powA > min ) )
                {
                    float32v gen = Gen( seedV, xPos, yPos, zPos );
                    out += gen * powA;
                }

                if( FS_AnyMask_bool( powB > min ) )
                {
                    float32v gen = genBFS->Gen( seedV, xPos, yPos, zPos );
                    out += gen * powB;
                }

                float32v factor = FS_Min_f32( FS_Max_f32( ( domain + over ) - biomPowerV, min ), over ) / over;

                FS_Store_f32( &noiseOut[index], ( FS_Load_f32( &noiseOut[index] ) + ( out * factor ) ) );

                int32v biomSwitchV = FS_Load_i32( &biomSwitch[index] );
                int32v biomListV = FS_Load_i32( &biomList[index] );
                mask32v mask = domain > biomPowerV;

                FS_Store_f32( &biomPower[index], FS_Select_f32( mask, domain, biomPowerV ) );
                FS_Store_i32( &biomSwitch[index], FS_Select_i32( mask, FS_Select_i32( basePow > min, p, m ), biomSwitchV ) );
                FS_Store_i32( &biomList[index], FS_Select_i32( mask, biomIndexV, biomListV ) );
            }

            index += FS_Size_32();
        }
    }

    FastNoise::OutputMinMax GenUniformGrid4D( float* noiseOut, int xStart, int yStart, int zStart, int wStart, int xSize, int ySize, int zSize, int wSize, float frequency, int seed ) const final
    {
        float32v min( INFINITY );
        float32v max( -INFINITY );

        int32v xIdx( xStart );
        int32v yIdx( yStart );
        int32v zIdx( zStart );
        int32v wIdx( wStart );

        float32v freqV( frequency );

        int32v xSizeV( xSize );
        int32v xMax = xSizeV + xIdx + int32v( -1 );
        int32v ySizeV( ySize );
        int32v yMax = ySizeV + yIdx + int32v( -1 );
        int32v zSizeV( zSize );
        int32v zMax = zSizeV + zIdx + int32v( -1 );

        intptr_t totalValues = xSize * ySize * zSize * wSize;
        intptr_t index = 0;

        xIdx += int32v::FS_Incremented();

        AxisReset<true>( xIdx, yIdx, xMax, xSizeV, xSize );
        AxisReset<true>( yIdx, zIdx, yMax, ySizeV, xSize * ySize );
        AxisReset<true>( zIdx, wIdx, zMax, zSizeV, xSize * ySize * zSize );

        while( index < totalValues - (intptr_t)FS_Size_32() )
        {
            float32v xPos = FS_Converti32_f32( xIdx ) * freqV;
            float32v yPos = FS_Converti32_f32( yIdx ) * freqV;
            float32v zPos = FS_Converti32_f32( zIdx ) * freqV;
            float32v wPos = FS_Converti32_f32( wIdx ) * freqV;

            float32v gen = Gen( int32v( seed ), xPos, yPos, zPos, wPos );
            FS_Store_f32( &noiseOut[index], gen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS_Min_f32( min, gen );
            max = FS_Max_f32( max, gen );
#endif

            index += FS_Size_32();
            xIdx += int32v( FS_Size_32() );

            AxisReset<false>( xIdx, yIdx, xMax, xSizeV, xSize );
            AxisReset<false>( yIdx, zIdx, yMax, ySizeV, xSize * ySize );
            AxisReset<false>( zIdx, wIdx, zMax, zSizeV, xSize * ySize * zSize );
        }

        float32v xPos = FS_Converti32_f32( xIdx ) * freqV;
        float32v yPos = FS_Converti32_f32( yIdx ) * freqV;
        float32v zPos = FS_Converti32_f32( zIdx ) * freqV;
        float32v wPos = FS_Converti32_f32( wIdx ) * freqV;

        float32v gen = Gen( int32v( seed ), xPos, yPos, zPos, wPos );

        return DoRemaining( noiseOut, totalValues, index, min, max, gen );
    }

    FastNoise::OutputMinMax GenPositionArray2D( float* noiseOut, int count, const float* xPosArray, const float* yPosArray, float xOffset, float yOffset, int seed ) const final
    {
        float32v min( INFINITY );
        float32v max( -INFINITY );

        intptr_t index = 0;
        while( index < count - (intptr_t)FS_Size_32() )
        {
            float32v xPos = float32v( xOffset ) + FS_Load_f32( &xPosArray[index] );
            float32v yPos = float32v( yOffset ) + FS_Load_f32( &yPosArray[index] );

            float32v gen = Gen( int32v( seed ), xPos, yPos );
            FS_Store_f32( &noiseOut[index], gen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS_Min_f32( min, gen );
            max = FS_Max_f32( max, gen );
#endif
            index += FS_Size_32();
        }

        float32v xPos = float32v( xOffset ) + FS_Load_f32( &xPosArray[index] );
        float32v yPos = float32v( yOffset ) + FS_Load_f32( &yPosArray[index] );

        float32v gen = Gen( int32v( seed ), xPos, yPos );

        return DoRemaining( noiseOut, count, index, min, max, gen );
    }

    FastNoise::OutputMinMax GenPositionArray3D( float* noiseOut, int count, const float* xPosArray, const float* yPosArray, const float* zPosArray, float xOffset, float yOffset, float zOffset, int seed ) const final
    {
        float32v min( INFINITY );
        float32v max( -INFINITY );

        intptr_t index = 0;
        while( index < count - (intptr_t)FS_Size_32() )
        {
            float32v xPos = float32v( xOffset ) + FS_Load_f32( &xPosArray[index] );
            float32v yPos = float32v( yOffset ) + FS_Load_f32( &yPosArray[index] );
            float32v zPos = float32v( zOffset ) + FS_Load_f32( &zPosArray[index] );

            float32v gen = Gen( int32v( seed ), xPos, yPos, zPos );
            FS_Store_f32( &noiseOut[index], gen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS_Min_f32( min, gen );
            max = FS_Max_f32( max, gen );
#endif
            index += FS_Size_32();
        }

        float32v xPos = float32v( xOffset ) + FS_Load_f32( &xPosArray[index] );
        float32v yPos = float32v( yOffset ) + FS_Load_f32( &yPosArray[index] );
        float32v zPos = float32v( zOffset ) + FS_Load_f32( &zPosArray[index] );

        float32v gen = Gen( int32v( seed ), xPos, yPos, zPos );

        return DoRemaining( noiseOut, count, index, min, max, gen );
    }

    FastNoise::OutputMinMax GenPositionArray4D( float* noiseOut, int count, const float* xPosArray, const float* yPosArray, const float* zPosArray, const float* wPosArray, float xOffset, float yOffset, float zOffset, float wOffset, int seed ) const final
    {
        float32v min( INFINITY );
        float32v max( -INFINITY );

        intptr_t index = 0;
        while( index < count - (intptr_t)FS_Size_32() )
        {
            float32v xPos = float32v( xOffset ) + FS_Load_f32( &xPosArray[index] );
            float32v yPos = float32v( yOffset ) + FS_Load_f32( &yPosArray[index] );
            float32v zPos = float32v( zOffset ) + FS_Load_f32( &zPosArray[index] );
            float32v wPos = float32v( wOffset ) + FS_Load_f32( &wPosArray[index] );

            float32v gen = Gen( int32v( seed ), xPos, yPos, zPos, wPos );
            FS_Store_f32( &noiseOut[index], gen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS_Min_f32( min, gen );
            max = FS_Max_f32( max, gen );
#endif
            index += FS_Size_32();
        }

        float32v xPos = float32v( xOffset ) + FS_Load_f32( &xPosArray[index] );
        float32v yPos = float32v( yOffset ) + FS_Load_f32( &yPosArray[index] );
        float32v zPos = float32v( zOffset ) + FS_Load_f32( &zPosArray[index] );
        float32v wPos = float32v( wOffset ) + FS_Load_f32( &wPosArray[index] );

        float32v gen = Gen( int32v( seed ), xPos, yPos, zPos, wPos );

        return DoRemaining( noiseOut, count, index, min, max, gen );
    }

    float GenSingle2D( float x, float y, int seed ) const final
    {
        return FS_Extract0_f32( Gen( int32v( seed ), float32v( x ), float32v( y ) ) );
    }

    float GenSingle3D( float x, float y, float z, int seed ) const final
    {
        return FS_Extract0_f32( Gen( int32v( seed ), float32v( x ), float32v( y ), float32v( z ) ) );
    }

    float GenSingle4D( float x, float y, float z, float w, int seed ) const final
    {
        return FS_Extract0_f32( Gen( int32v( seed ), float32v( x ), float32v( y ), float32v( z ), float32v( w ) ) );
    }

    FastNoise::OutputMinMax GenTileable2D( float* noiseOut, int xSize, int ySize, float frequency, int seed ) const final
    {
        float32v min( INFINITY );
        float32v max( -INFINITY );

        int32v xIdx( 0 );
        int32v yIdx( 0 );

        int32v xSizeV( xSize );
        int32v ySizeV( ySize );
        int32v xMax = xSizeV + xIdx + int32v( -1 );

        intptr_t totalValues = xSize * ySize;
        intptr_t index = 0;

        float pi2Recip( 0.15915493667f );
        float xSizePi = (float)xSize * pi2Recip;
        float ySizePi = (float)ySize * pi2Recip;
        float32v xFreq = float32v( frequency * xSizePi );
        float32v yFreq = float32v( frequency * ySizePi );
        float32v xMul = float32v( 1 / xSizePi );
        float32v yMul = float32v( 1 / ySizePi );

        xIdx += int32v::FS_Incremented();

        AxisReset<true>( xIdx, yIdx, xMax, xSizeV, xSize );

        while( index < totalValues - (intptr_t)FS_Size_32() )
        {
            float32v xF = FS_Converti32_f32( xIdx ) * xMul;
            float32v yF = FS_Converti32_f32( yIdx ) * yMul;

            float32v xPos = FS_Cos_f32( xF ) * xFreq;
            float32v yPos = FS_Cos_f32( yF ) * yFreq;
            float32v zPos = FS_Sin_f32( xF ) * xFreq;
            float32v wPos = FS_Sin_f32( yF ) * yFreq;

            float32v gen = Gen( int32v( seed ), xPos, yPos, zPos, wPos );
            FS_Store_f32( &noiseOut[index], gen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS_Min_f32( min, gen );
            max = FS_Max_f32( max, gen );
#endif

            index += FS_Size_32();
            xIdx += int32v( FS_Size_32() );

            AxisReset<false>( xIdx, yIdx, xMax, xSizeV, xSize );
        }

        float32v xF = FS_Converti32_f32( xIdx ) * xMul;
        float32v yF = FS_Converti32_f32( yIdx ) * yMul;

        float32v xPos = FS_Cos_f32( xF ) * xFreq;
        float32v yPos = FS_Cos_f32( yF ) * yFreq;
        float32v zPos = FS_Sin_f32( xF ) * xFreq;
        float32v wPos = FS_Sin_f32( yF ) * yFreq;

        float32v gen = Gen( int32v( seed ), xPos, yPos, zPos, wPos );

        return DoRemaining( noiseOut, totalValues, index, min, max, gen );
    }

private:
    template<bool INITIAL>
    static FS_INLINE void AxisReset( int32v& aIdx, int32v& bIdx, int32v aMax, int32v aSize, size_t aStep )
    {
        for( size_t resetLoop = INITIAL ? aStep : 0; resetLoop < FS_Size_32(); resetLoop += aStep )
        {
            mask32v aReset = aIdx > aMax;
            bIdx = FS_MaskedIncrement_i32( bIdx, aReset );
            aIdx = FS_MaskedSub_i32( aIdx, aSize, aReset );
        }
    }

    static FS_INLINE FastNoise::OutputMinMax DoRemaining( float* noiseOut, intptr_t totalValues, intptr_t index, float32v min, float32v max, float32v finalGen )
    {
        FastNoise::OutputMinMax minMax;
        intptr_t remaining = totalValues - index;

        if( remaining == (intptr_t)FS_Size_32() )
        {
            FS_Store_f32( &noiseOut[index], finalGen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS_Min_f32( min, finalGen );
            max = FS_Max_f32( max, finalGen );
#endif
        }
        else
        {
            std::memcpy( &noiseOut[index], &finalGen, remaining * sizeof( float ) );

#if FASTNOISE_CALC_MIN_MAX
            do
            {
                minMax << noiseOut[index];
            }
            while( ++index < totalValues );
#endif
        }

#if FASTNOISE_CALC_MIN_MAX
        float* minP = reinterpret_cast<float*>(&min);
        float* maxP = reinterpret_cast<float*>(&max);
        for( size_t i = 0; i < FS_Size_32(); i++ )
        {
            minMax << FastNoise::OutputMinMax{ minP[i], maxP[i] };
        }
#endif

        return minMax;
    }
};
