#pragma once
#include "Generator.h"

namespace FastNoise
{
    /** @brief How a sample between grid cells is resolved. */
    enum class HeightMapInterpolation
    {
        Nearest, ///< Take the nearest cell. One memory tap per sample.
        Linear,  ///< Blend neighbouring cells. 4 taps for a 2D grid, 8 for a 3D one.
    };

    constexpr static const char* kHeightMapInterpolation_Strings[] =
    {
        "Nearest",
        "Linear",
    };

    /** @brief What happens to samples outside the grid, per axis. */
    enum class HeightMapTiling
    {
        ClampEdge, ///< The edge cell extends outwards forever.
        Repeat,    ///< The grid repeats. Cell `n-1` sits next to cell `0`, so their contents must match or a hard seam appears.
        Mirror,    ///< The grid repeats mirrored. Always seamless, at the cost of visible symmetry axes. The edge cell is doubled (GL_MIRRORED_REPEAT).
        Border,    ///< Everything outside reads "Border Value". This is how a finite island is placed in an otherwise procedural world.
    };

    constexpr static const char* kHeightMapTiling_Strings[] =
    {
        "Clamp Edge",
        "Repeat",
        "Mirror",
        "Border",
    };

    /** @brief Storage type of one grid cell.
     *
     *  Not reflected: it describes the data, not the graph. Keeping the source
     *  type instead of widening everything to float halves or quarters the
     *  bandwidth, which is what decides whether a grid stays in cache.
     */
    enum class HeightMapElement
    {
        UInt8,
        UInt16,
        Float32,
    };

    /** @brief Samples a caller-supplied grid of values instead of computing noise.
     *
     *  The grid itself is attached with SetGrid() and is deliberately NOT part of
     *  the reflected metadata: FastNoise2 member variables can only be float, int
     *  or enum, and a grid is a pointer plus its extents. What IS reflected is
     *  only what a graph author should be able to override — interpolation, the
     *  per-axis tiling mode, and the value read outside the grid.
     *
     *  The value leaves this node exactly as it was stored, apart from the
     *  unavoidable decode of the storage format (`value * scale + bias`, both
     *  folded into two constants by SetGrid). Nothing here adapts the value to a
     *  world, a height or a density — combine it with other nodes for that.
     *
     *  With no grid attached every sample is outside it, so every sample returns
     *  "Border Value". That is deterministic, finite, and reads no memory.
     */
    class HeightMap : public virtual Generator
    {
    public:
        const Metadata& GetMetadata() const override;

        void SetInterpolation( HeightMapInterpolation value ) { mInterpolation = value; }
        void SetTilingU( HeightMapTiling value ) { mTiling[0] = value; }
        void SetTilingV( HeightMapTiling value ) { mTiling[1] = value; }
        void SetTilingW( HeightMapTiling value ) { mTiling[2] = value; }
        void SetBorderValue( float value ) { mBorderValue = value; RefreshBorderRaw(); }

        /** @brief Attaches the grid this node samples.
         *
         *  Inline on purpose: the caller owns the memory and the maths, so nothing
         *  about a grid ever crosses the library boundary as an allocation.
         *
         *  @param data        First cell. Layout is U-major: `((w * sizeV) + v) * sizeU + u`.
         *                     U must be the axis the caller iterates fastest, or every
         *                     neighbour tap costs a cache miss.
         *                     Pass nullptr to detach (see the class description).
         *  @param element     Storage type of one cell.
         *  @param dimensions  2 or 3. A 2D grid is constant along the third axis and
         *                     is never sampled there, so it costs 4 taps instead of 8.
         *  @param sizeU/V/W   Cell counts. sizeW is ignored when dimensions == 2.
         *  @param axisMap     Which incoming position axis feeds U, V and W
         *                     (0 = x, 1 = y, 2 = z). Only the first `dimensions`
         *                     entries are read.
         *  @param origin      Position, in the same units the caller passes to Gen(),
         *                     of the CENTRE of cell 0 on each of U, V, W.
         *  @param cell        Size of one cell in those same units, per axis. Must be
         *                     non-zero; the reciprocal is taken here so the kernel
         *                     never divides.
         *  @param valueScale  Multiplied onto the raw stored number.
         *  @param valueBias   Added afterwards. Together these turn a raw U8/U16 back
         *                     into the number it stood for, and carry any caller-side
         *                     factor for free.
         */
        void SetGrid(
            const void* data, HeightMapElement element, int dimensions,
            int sizeU, int sizeV, int sizeW,
            const int* axisMap,
            const float* origin, const float* cell,
            float valueScale, float valueBias )
        {
            mData = data;
            mElement = element;
            mDimensions = dimensions < 3 ? 2 : 3;

            mSize[0] = sizeU;
            mSize[1] = sizeV;
            mSize[2] = mDimensions == 3 ? sizeW : 1;

            mStrideV = sizeU;
            mStrideW = sizeU * sizeV;

            mValueScale = valueScale;
            mValueBias = valueBias;

            for( int i = 0; i < 3; i++ )
            {
                mAxis[i] = ( i < mDimensions && axisMap[i] >= 0 && axisMap[i] < 3 ) ? axisMap[i] : i;

                // Everything the kernel needs, precomputed: a grid position is one
                // fused multiply-add, and no tiling mode reaches a divide.
                const float cellSize = ( i < mDimensions && cell[i] != 0.0f ) ? cell[i] : 1.0f;
                mInvCell[i] = 1.0f / cellSize;
                mBias[i] = -origin[i] * mInvCell[i];

                const float n = (float)( mSize[i] > 0 ? mSize[i] : 1 );
                mSizeF[i] = n;
                mInvSize[i] = 1.0f / n;
                mInvSize2[i] = 0.5f / n;
                mMirrorFold[i] = 2.0f * n - 1.0f;
                mMaxIndex[i] = mSize[i] > 0 ? mSize[i] - 1 : 0;
            }

            // The scale and bias just installed are the other half of the border
            // conversion, and SetGrid and SetBorderValue may be called in either
            // order, so both ends refresh it.
            RefreshBorderRaw();
        }

        /** The border expressed in stored units, so it survives the decode every
         *  tap goes through. Exact for the identity decode a float map installs. */
        void RefreshBorderRaw()
        {
            mBorderRaw = ( mValueScale != 0.0f )
                ? ( mBorderValue - mValueBias ) / mValueScale
                : mBorderValue;
        }

        /** @brief Detaches the grid. Every sample then returns "Border Value". */
        void ClearGrid()
        {
            const int identity[3] = { 0, 1, 2 };
            const float zero[3] = { 0.0f, 0.0f, 0.0f };
            const float one[3] = { 1.0f, 1.0f, 1.0f };
            SetGrid( nullptr, HeightMapElement::Float32, 3, 0, 0, 0, identity, zero, one, 1.0f, 0.0f );
        }

        bool HasGrid() const { return mData != nullptr; }

    protected:
        // --- reflected -------------------------------------------------------
        HeightMapInterpolation mInterpolation = HeightMapInterpolation::Linear;
        HeightMapTiling mTiling[3] = { HeightMapTiling::ClampEdge, HeightMapTiling::ClampEdge, HeightMapTiling::ClampEdge };
        float mBorderValue = 0.0f;

        // --- derived from mBorderValue and the grid's decode ------------------
        float mBorderRaw = 0.0f;

        // --- attached by SetGrid --------------------------------------------
        const void* mData = nullptr;
        HeightMapElement mElement = HeightMapElement::Float32;
        int mDimensions = 3;

        int mAxis[3] = { 0, 1, 2 };
        int mSize[3] = { 0, 0, 0 };
        int mMaxIndex[3] = { 0, 0, 0 };
        int mStrideV = 0;
        int mStrideW = 0;

        float mInvCell[3] = { 1.0f, 1.0f, 1.0f };
        float mBias[3] = { 0.0f, 0.0f, 0.0f };
        float mSizeF[3] = { 1.0f, 1.0f, 1.0f };
        float mInvSize[3] = { 1.0f, 1.0f, 1.0f };
        float mInvSize2[3] = { 0.5f, 0.5f, 0.5f };
        float mMirrorFold[3] = { 1.0f, 1.0f, 1.0f };

        float mValueScale = 1.0f;
        float mValueBias = 0.0f;

        template<typename T>
        friend struct MetadataT;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<HeightMap> : MetadataT<Generator>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT()
        {
            groups.push_back( "Basic Generators" );

            this->AddVariableEnum( { "Interpolation", "Nearest costs one memory tap per sample, Linear costs 4 (2D grid) or 8 (3D grid)" },
                HeightMapInterpolation::Linear, &HeightMap::SetInterpolation, kHeightMapInterpolation_Strings );

            this->AddVariableEnum( { "U Tiling", "What happens outside the grid along its first axis" },
                HeightMapTiling::ClampEdge, &HeightMap::SetTilingU, kHeightMapTiling_Strings );
            this->AddVariableEnum( { "V Tiling", "What happens outside the grid along its second axis" },
                HeightMapTiling::ClampEdge, &HeightMap::SetTilingV, kHeightMapTiling_Strings );
            this->AddVariableEnum( { "W Tiling", "What happens outside the grid along its third axis\nIgnored by a 2D grid" },
                HeightMapTiling::ClampEdge, &HeightMap::SetTilingW, kHeightMapTiling_Strings );

            this->AddVariable( { "Border Value", "Output for samples outside the grid where tiling is set to Border\nAlso the output everywhere when no grid is attached" },
                0.0f, &HeightMap::SetBorderValue );

            description =
                "Samples a grid of stored values supplied by the host application\n"
                "The stored value is returned as-is, apart from decoding its storage format\n"
                "Mirror doubles the edge cell, Repeat puts cell n-1 next to cell 0";
        }
    };
#endif
}
