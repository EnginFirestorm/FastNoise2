#include "HeightMap.h"
#include "Utils.inl"

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<HeightMap, SIMD> final : public virtual HeightMap, public DispatchClass<Generator, SIMD>
{
    static constexpr int kLanes = int32v::ElementCount;
    static constexpr int kMaxTaps = 8;

    FASTNOISE_IMPL_GEN_T;

    /** Wraps a grid coordinate into the grid, entirely in float.
     *
     *  Integer division does not exist in FastSIMD and the reciprocals are
     *  precomputed by SetGrid, so every mode here is multiplies and a floor.
     *  ClampEdge and Border return the coordinate untouched: the unconditional
     *  integer clamp in ToIndex covers ClampEdge, and Border is resolved by the
     *  out-of-bounds flag instead. */
    FS_FORCEINLINE float32v ApplyTiling( float32v f, int a ) const
    {
        switch( mTiling[a] )
        {
        case HeightMapTiling::Repeat:
            return FS::FMulAdd( float32v( -mSizeF[a] ), FS::Floor( f * float32v( mInvSize[a] ) ), f );

        case HeightMapTiling::Mirror:
        {
            // Period 2n, then fold the upper half back: 2n-1-r. Doubles the edge
            // cell, which is what GL_MIRRORED_REPEAT does and why it is seamless.
            const float32v r = FS::FMulAdd( float32v( -2.0f * mSizeF[a] ), FS::Floor( f * float32v( mInvSize2[a] ) ), f );
            return FS::Min( r, float32v( mMirrorFold[a] ) - r );
        }

        case HeightMapTiling::ClampEdge:
        case HeightMapTiling::Border:
        default:
            return f;
        }
    }

    /** Turns a wrapped coordinate into a cell index that is ALWAYS inside the
     *  buffer. The clamp is memory safety, not tidiness: a NaN coordinate becomes
     *  INT_MIN when converted on x86 and saturates on NEON, and either one indexes
     *  outside an allocation the library cannot diagnose. */
    FS_FORCEINLINE int32v ToIndex( float32v f, int a ) const
    {
        const int32v i = FS::Convert<std::int32_t>( f );
        return FS::Min( FS::Max( i, int32v( 0 ) ), int32v( mMaxIndex[a] ) );
    }

    /** 1.0 where this coordinate lies outside the grid, 0.0 elsewhere. Only ever
     *  non-zero for an axis whose tiling is Border. */
    FS_FORCEINLINE float32v OutsideFlag( float32v f, int a ) const
    {
        if( mTiling[a] != HeightMapTiling::Border )
        {
            return float32v( 0.0f );
        }
        const auto m = ( f < float32v( 0.0f ) ) | ( f > float32v( mSizeF[a] - 1.0f ) );
        return FS::Select( m, float32v( 1.0f ), float32v( 0.0f ) );
    }

    /** Fills the index (and out-of-bounds) lanes for one axis.
     *  `frac` is only written for Linear. */
    FS_FORCEINLINE void PrepareAxis( float32v g, int a, bool linear, bool border,
        std::int32_t* idx0, std::int32_t* idx1,
        float* oob0, float* oob1, float32v& frac ) const
    {
        float32v f0;
        if( linear )
        {
            f0 = FS::Floor( g );
            frac = g - f0;
        }
        else
        {
            f0 = FS::Floor( g + float32v( 0.5f ) );
        }

        FS::Store( idx0, ToIndex( ApplyTiling( f0, a ), a ) );
        if( border )
        {
            FS::Store( oob0, OutsideFlag( f0, a ) );
        }

        if( linear )
        {
            const float32v f1 = f0 + float32v( 1.0f );
            FS::Store( idx1, ToIndex( ApplyTiling( f1, a ), a ) );
            if( border )
            {
                FS::Store( oob1, OutsideFlag( f1, a ) );
            }
        }
    }

    /** The scalar half of the fetch: address arithmetic and the loads themselves.
     *
     *  FastSIMD has no gather, so the values have to be picked up one at a time.
     *  The addresses are computed here rather than in SIMD because an int32v
     *  multiply is emulated through shuffles on SSE2 — and computing them inside
     *  this loop means the index never has to be written to memory and read back.
     *
     *  `oob` is null unless some axis tiles as Border. */
    template<typename T, bool IS_3D, bool BORDER>
    FS_FORCEINLINE void GatherTaps( float* out, const std::int32_t ( &axisIdx )[3][2][kLanes],
        const float ( &axisOob )[3][2][kLanes], int tapCount ) const
    {
        const T* src = (const T*)mData;
        const int strideV = mStrideV;
        const int strideW = mStrideW;

        // The border enters the tap array beside stored cells, and every tap leaves
        // this function through the same decode (value scale and bias) at the end of
        // GenT. So what goes in here is the border expressed in STORED units, not
        // the value the caller asked for — mBorderRaw is that, precomputed.
        //
        // Using mBorderValue directly was wrong in a way only integer storage showed:
        // for Float32 the decode is the identity and the two are equal, but a U8 or
        // U16 map decodes with the range it was packed against, and a border of +1
        // came out as 1*scale + bias. With a range that spans negative values that
        // lands on the same side of zero as -1 does, so a map's outside read as
        // solid whichever sign the customer set — and no test caught it, because
        // every tiling test used a float map.
        const float borderValue = mBorderRaw;

        for( int t = 0; t < tapCount; t++ )
        {
            const std::int32_t* iu = axisIdx[0][t & 1];
            const std::int32_t* iv = axisIdx[1][( t >> 1 ) & 1];
            const std::int32_t* iw = IS_3D ? axisIdx[2][( t >> 2 ) & 1] : nullptr;

            float* dst = out + t * kLanes;

            if constexpr( BORDER )
            {
                const float* ou = axisOob[0][t & 1];
                const float* ov = axisOob[1][( t >> 1 ) & 1];
                const float* ow = IS_3D ? axisOob[2][( t >> 2 ) & 1] : nullptr;

                for( int l = 0; l < kLanes; l++ )
                {
                    const bool outside = ou[l] != 0.0f || ov[l] != 0.0f || ( IS_3D && ow[l] != 0.0f );
                    const int idx = IS_3D ? ( iw[l] * strideW + iv[l] * strideV + iu[l] )
                                          : ( iv[l] * strideV + iu[l] );
                    dst[l] = outside ? borderValue : (float)src[idx];
                }
            }
            else
            {
                for( int l = 0; l < kLanes; l++ )
                {
                    const int idx = IS_3D ? ( iw[l] * strideW + iv[l] * strideV + iu[l] )
                                          : ( iv[l] * strideV + iu[l] );
                    dst[l] = (float)src[idx];
                }
            }
        }
    }

    /** Element type is the outermost switch so the load loop above sees a concrete
     *  type and no branch. */
    template<bool IS_3D, bool BORDER>
    FS_FORCEINLINE void GatherByElement( float* out, const std::int32_t ( &axisIdx )[3][2][kLanes],
        const float ( &axisOob )[3][2][kLanes], int tapCount ) const
    {
        switch( mElement )
        {
        case HeightMapElement::UInt8:  GatherTaps<std::uint8_t, IS_3D, BORDER>( out, axisIdx, axisOob, tapCount ); break;
        case HeightMapElement::UInt16: GatherTaps<std::uint16_t, IS_3D, BORDER>( out, axisIdx, axisOob, tapCount ); break;
        case HeightMapElement::Float32:
        default:                       GatherTaps<float, IS_3D, BORDER>( out, axisIdx, axisOob, tapCount ); break;
        }
    }

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        // No grid means zero extent, so every sample is outside it. Deterministic,
        // finite, and it reads no memory.
        if( !mData )
        {
            return float32v( mBorderValue );
        }

        // Gen() arrives with 2, 3 or 4 positions; the axis map indexes into this.
        float32v axisPos[4] = { float32v( 0.0f ), float32v( 0.0f ), float32v( 0.0f ), float32v( 0.0f ) };
        {
            size_t fill = 0;
            ( ( axisPos[fill++] = pos ), ... );
        }

        const bool is3D = mDimensions == 3;
        const int axisCount = is3D ? 3 : 2;
        const bool linear = mInterpolation == HeightMapInterpolation::Linear;
        const bool border = mTiling[0] == HeightMapTiling::Border
                         || mTiling[1] == HeightMapTiling::Border
                         || ( is3D && mTiling[2] == HeightMapTiling::Border );

        // Deliberately not zero-initialised: only the entries an axis actually
        // writes are ever read back, and clearing them would cost a memset of the
        // whole thing on every single call.
        std::int32_t axisIdx[3][2][kLanes];
        float axisOob[3][2][kLanes];
        float32v frac[3];

        for( int a = 0; a < axisCount; a++ )
        {
            // One fused multiply-add per axis: SetGrid folded origin and cell size
            // into these two constants.
            const float32v g = FS::FMulAdd( axisPos[mAxis[a]], float32v( mInvCell[a] ), float32v( mBias[a] ) );
            PrepareAxis( g, a, linear, border, axisIdx[a][0], axisIdx[a][1], axisOob[a][0], axisOob[a][1], frac[a] );
        }

        // Tap order is u fastest, then v, then w — the same order the grid is laid
        // out in, so neighbouring taps stay in one cache line.
        const int tapCount = linear ? ( is3D ? 8 : 4 ) : 1;
        float tapVal[kMaxTaps * kLanes];

        if( is3D )
        {
            if( border ) GatherByElement<true, true>( tapVal, axisIdx, axisOob, tapCount );
            else         GatherByElement<true, false>( tapVal, axisIdx, axisOob, tapCount );
        }
        else
        {
            if( border ) GatherByElement<false, true>( tapVal, axisIdx, axisOob, tapCount );
            else         GatherByElement<false, false>( tapVal, axisIdx, axisOob, tapCount );
        }

        float32v result;
        if( !linear )
        {
            result = FS::Load<float32v>( tapVal );
        }
        else
        {
            const float32v t00 = Lerp( FS::Load<float32v>( tapVal + 0 * kLanes ), FS::Load<float32v>( tapVal + 1 * kLanes ), frac[0] );
            const float32v t01 = Lerp( FS::Load<float32v>( tapVal + 2 * kLanes ), FS::Load<float32v>( tapVal + 3 * kLanes ), frac[0] );
            const float32v v0 = Lerp( t00, t01, frac[1] );

            if( !is3D )
            {
                result = v0;
            }
            else
            {
                const float32v t10 = Lerp( FS::Load<float32v>( tapVal + 4 * kLanes ), FS::Load<float32v>( tapVal + 5 * kLanes ), frac[0] );
                const float32v t11 = Lerp( FS::Load<float32v>( tapVal + 6 * kLanes ), FS::Load<float32v>( tapVal + 7 * kLanes ), frac[0] );
                const float32v v1 = Lerp( t10, t11, frac[1] );
                result = Lerp( v0, v1, frac[2] );
            }
        }

        // The only arithmetic applied to the value: undo the storage format. Any
        // caller-side factor was folded into these two constants, so it is free.
        return FS::FMulAdd( result, float32v( mValueScale ), float32v( mValueBias ) );
    }
};
