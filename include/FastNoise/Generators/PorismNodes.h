#pragma once
#include <cmath>

#include "Generator.h"

// Porism additions on top of upstream FastNoise2 v1.1.1.
//
// Every node here exists for one of two reasons:
//  - it fuses a multi-node pattern that appears throughout the Porism noise
//    corpus into one node (BoxDomain), with a Strict mode that reproduces the
//    primitive chain bit-for-bit, or
//  - it fills a gap other noise systems cover (Billow, Musgrave hybrid
//    multifractal, Schlick bias/gain, curves, clamp, scale+bias, threshold).
//
// Wire-format rule: these classes register at the END of the node list in
// FastSIMD_Build.inl. Their internal member order is wire contract too - new
// members go last, never in between.

namespace FastNoise
{
    /** @brief FBm fractal over the absolute source value, remapped to [-1, 1].
     *
     *  Classic "billow" look: rounded, cloud-like ridges where FBm is smooth.
     *  Identical to FractalFBm except every octave samples
     *  `Abs(source) * 2 - 1` instead of the raw source.
     */
    class FractalBillow : public virtual Fractal<>
    {
    public:
        const Metadata& GetMetadata() const override;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<FractalBillow> : MetadataT<Fractal<>>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT() : MetadataT<Fractal<>>()
        {
            description =
                "Fractional Brownian Motion over the absolute source value\n"
                "Each octave contributes Abs(source) * 2 - 1, producing rounded\n"
                "billowing shapes (clouds, puffy hills) where FBm stays smooth";
        }
    };
#endif

    /** @brief FBm whose output is divided by the amplitude it actually summed.
     *
     *  Upstream FractalFBm reaches `1 + gain + gain^2 + ...` for as many terms
     *  as there are octaves. This node tracks that sum per sample and divides,
     *  so the output stays in the source's range regardless of Gain/Octaves -
     *  including a Gain driven by another node, which no constant multiply can
     *  normalise.
     */
    class FractalFBmNormalized : public virtual Fractal<>
    {
    public:
        const Metadata& GetMetadata() const override;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<FractalFBmNormalized> : MetadataT<Fractal<>>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT() : MetadataT<Fractal<>>()
        {
            description =
                "FractalFBm divided by its summed octave amplitude\n"
                "Output stays within the source's own range no matter how Gain\n"
                "and Octaves are set, even when Gain varies across the world";
        }
    };
#endif

    /** @brief Musgrave hybrid multifractal.
     *
     *  Octaves are weighted by the running product of previous octave values,
     *  so valleys stay smooth while peaks accumulate detail. Spectral octave
     *  weights `lacunarity^(-i * H)` are precomputed on the setters - H,
     *  Offset, Octaves and Lacunarity are therefore plain values, not inputs.
     */
    class FractalHybridMulti : public virtual Generator
    {
    public:
        static constexpr int kMaxOctaves = 16;

        const Metadata& GetMetadata() const override;

        void SetSource( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mSource, gen ); }
        void SetOffset( float value ) { mOffset = value; }
        void SetH( float value ) { mH = value; RecalculateWeights(); }
        void SetOctaveCount( int value ) { mOctaves = value; }
        void SetLacunarity( float value ) { mLacunarity = value; RecalculateWeights(); }

    protected:
        void RecalculateWeights()
        {
            for( int i = 0; i < kMaxOctaves; i++ )
            {
                mSpectralWeights[i] = std::pow( mLacunarity, -i * mH );
            }
        }

        GeneratorSource mSource;
        float mOffset = 0.7f;
        float mH = 0.25f;
        int   mOctaves = 4;
        float mLacunarity = 2.0f;
        float mSpectralWeights[kMaxOctaves] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

        template<typename T>
        friend struct MetadataT;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<FractalHybridMulti> : MetadataT<Generator>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT()
        {
            groups.push_back( "Fractal" );
            this->AddGeneratorSource( "Source", &FractalHybridMulti::SetSource );
            this->AddVariable( { "Offset",
                "Added to every octave before weighting\n"
                "Around 0.7 gives the classic Musgrave look" },
                0.7f, &FractalHybridMulti::SetOffset );
            this->AddVariable( { "H",
                "Spectral exponent - how fast octave amplitude falls off\n"
                "Octave i is weighted by Lacunarity^(-i * H)" },
                0.25f, &FractalHybridMulti::SetH );
            this->AddVariable( "Octaves", 4, &FractalHybridMulti::SetOctaveCount, 2, FractalHybridMulti::kMaxOctaves );
            this->AddVariable( "Lacunarity", 2.0f, &FractalHybridMulti::SetLacunarity );

            description =
                "Musgrave hybrid multifractal\n"
                "Each octave is scaled by the running product of previous octaves,\n"
                "keeping valley floors smooth while ridges accumulate detail\n"
                "The classic terrain multifractal from Texturing & Modeling";
        }
    };
#endif

    /** @brief Schlick bias/gain curve over a [-1, 1] signal.
     *
     *  Bias shifts where the midpoint of the range sits, Gain sharpens or
     *  flattens the transition around it. 0.5 / 0.5 is the identity. Both are
     *  clamped to (0, 1) exclusive on the setter because the Schlick formula
     *  divides by them; the clamp bound is 1e-6.
     */
    class BiasGain : public virtual Generator
    {
    public:
        const Metadata& GetMetadata() const override;

        void SetSource( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mSource, gen ); }
        void SetBias( float value )
        {
            mBias = ClampOpen01( value );
            mBiasK = 1.0f / mBias - 2.0f;
        }
        void SetGainAmount( float value )
        {
            mGainAmount = ClampOpen01( value );
            mGainK = 1.0f / mGainAmount - 2.0f;
        }

    protected:
        static float ClampOpen01( float v )
        {
            constexpr float kEdge = 1e-6f;
            return v < kEdge ? kEdge : ( v > 1.0f - kEdge ? 1.0f - kEdge : v );
        }

        GeneratorSource mSource;
        float mBias = 0.5f;
        float mGainAmount = 0.5f;
        float mBiasK = 0.0f;
        float mGainK = 0.0f;

        template<typename T>
        friend struct MetadataT;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<BiasGain> : MetadataT<Generator>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT()
        {
            groups.push_back( "Modifiers" );
            this->AddGeneratorSource( "Source", &BiasGain::SetSource );
            this->AddVariable( { "Bias",
                "Where the midpoint of the output range sits\n"
                "0.5 = unchanged, lower pulls values down, higher pushes them up" },
                0.5f, &BiasGain::SetBias, 0.0f, 1.0f, 0.001f );
            this->AddVariable( { "Gain",
                "Contrast around the midpoint\n"
                "0.5 = unchanged, lower flattens, higher sharpens" },
                0.5f, &BiasGain::SetGainAmount, 0.0f, 1.0f, 0.001f );

            description =
                "Schlick bias/gain curve for reshaping a [-1, 1] signal\n"
                "Bias moves the midpoint, Gain controls contrast around it\n"
                "0.5 / 0.5 leaves the input unchanged";
        }
    };
#endif

    /** @brief Hard two-way select: below the threshold one value, above the other. */
    class Threshold : public virtual Generator
    {
    public:
        const Metadata& GetMetadata() const override;

        void SetSource( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mSource, gen ); }
        void SetThreshold( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mThreshold, gen ); }
        void SetThreshold( float value ) { mThreshold = value; }
        void SetBelow( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mBelow, gen ); }
        void SetBelow( float value ) { mBelow = value; }
        void SetAbove( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mAbove, gen ); }
        void SetAbove( float value ) { mAbove = value; }

    protected:
        GeneratorSource mSource;
        HybridSource mThreshold = 0.0f;
        HybridSource mBelow = -1.0f;
        HybridSource mAbove = 1.0f;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<Threshold> : MetadataT<Generator>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT()
        {
            groups.push_back( "Blends" );
            this->AddGeneratorSource( "Source", &Threshold::SetSource );
            this->AddHybridSource( "Threshold", 0.0f, &Threshold::SetThreshold, &Threshold::SetThreshold );
            this->AddHybridSource( "Below", -1.0f, &Threshold::SetBelow, &Threshold::SetBelow );
            this->AddHybridSource( "Above", 1.0f, &Threshold::SetAbove, &Threshold::SetAbove );

            description =
                "Outputs Below where the source is under the threshold, Above otherwise\n"
                "A hard step - for a soft transition use Fade";
        }
    };
#endif

    /** @brief Clamps the source between two bounds. Equivalent to Min(Max(x, lo), hi). */
    class Clamp : public virtual Generator
    {
    public:
        const Metadata& GetMetadata() const override;

        void SetSource( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mSource, gen ); }
        void SetMin( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mMin, gen ); }
        void SetMin( float value ) { mMin = value; }
        void SetMax( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mMax, gen ); }
        void SetMax( float value ) { mMax = value; }

    protected:
        GeneratorSource mSource;
        HybridSource mMin = -1.0f;
        HybridSource mMax = 1.0f;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<Clamp> : MetadataT<Generator>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT()
        {
            groups.push_back( "Modifiers" );
            this->AddGeneratorSource( "Source", &Clamp::SetSource );
            this->AddHybridSource( "Min", -1.0f, &Clamp::SetMin, &Clamp::SetMin );
            this->AddHybridSource( "Max", 1.0f, &Clamp::SetMax, &Clamp::SetMax );

            description =
                "Clamps the source output between Min and Max\n"
                "Bit-identical to the Min(Max(source, min), max) node chain";
        }
    };
#endif

    /** @brief `source * scale + bias` as two separate operations.
     *
     *  Deliberately NOT a fused multiply-add: the two-op form is bit-identical
     *  to the Add(Multiply(x, scale), bias) chain it replaces.
     */
    class ScaleBias : public virtual Generator
    {
    public:
        const Metadata& GetMetadata() const override;

        void SetSource( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mSource, gen ); }
        void SetScale( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mScale, gen ); }
        void SetScale( float value ) { mScale = value; }
        void SetBias( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mBias, gen ); }
        void SetBias( float value ) { mBias = value; }

    protected:
        GeneratorSource mSource;
        HybridSource mScale = 1.0f;
        HybridSource mBias = 0.0f;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<ScaleBias> : MetadataT<Generator>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT()
        {
            groups.push_back( "Modifiers" );
            this->AddGeneratorSource( "Source", &ScaleBias::SetSource );
            this->AddHybridSource( "Scale", 1.0f, &ScaleBias::SetScale, &ScaleBias::SetScale );
            this->AddHybridSource( "Bias", 0.0f, &ScaleBias::SetBias, &ScaleBias::SetBias );

            description =
                "source * Scale + Bias\n"
                "Bit-identical to the Add(Multiply(source, scale), bias) chain";
        }
    };
#endif

    /** @brief Piecewise-linear response curve over up to 8 control points.
     *
     *  Points must be authored with strictly ascending In values; the node does
     *  not sort. Samples below the first point return the first Out, samples
     *  past the last active segment return the last Out. Point Count says how
     *  many of the 8 point slots are read.
     */
    class Curve : public virtual Generator
    {
    public:
        static constexpr int kMaxPoints = 8;

        const Metadata& GetMetadata() const override;

        void SetSource( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mSource, gen ); }
        void SetPointCount( int value ) { mPointCount = value; }

        void SetPointIn( int index, float value ) { mPointIn[index] = value; }
        void SetPointOut( int index, float value ) { mPointOut[index] = value; }

    protected:
        GeneratorSource mSource;
        int mPointCount = 2;
        float mPointIn[kMaxPoints] = { -1.0f, 1.0f, 0, 0, 0, 0, 0, 0 };
        float mPointOut[kMaxPoints] = { -1.0f, 1.0f, 0, 0, 0, 0, 0, 0 };

        template<typename T>
        friend struct MetadataT;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<Curve> : MetadataT<Generator>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT()
        {
            groups.push_back( "Modifiers" );
            this->AddGeneratorSource( "Source", &Curve::SetSource );
            this->AddVariable( { "Point Count", "How many of the 8 point slots are used" },
                2, &Curve::SetPointCount, 2, Curve::kMaxPoints );

            // One In/Out pair per slot. Registered as indexed setters so the
            // member count stays within the wire format's 31-per-kind token cap.
            static const char* kInNames[Curve::kMaxPoints] =
            {
                "Point 0 In", "Point 1 In", "Point 2 In", "Point 3 In",
                "Point 4 In", "Point 5 In", "Point 6 In", "Point 7 In",
            };
            static const char* kOutNames[Curve::kMaxPoints] =
            {
                "Point 0 Out", "Point 1 Out", "Point 2 Out", "Point 3 Out",
                "Point 4 Out", "Point 5 Out", "Point 6 Out", "Point 7 Out",
            };
            const float kDefaults[Curve::kMaxPoints] = { -1.0f, 1.0f, 0, 0, 0, 0, 0, 0 };

            for( int i = 0; i < Curve::kMaxPoints; i++ )
            {
                this->AddVariable( { kInNames[i], "Input position of this control point (ascending)" },
                    kDefaults[i],
                    [i]( Curve* p, float f ) { p->mPointIn[i] = f; } );
                this->AddVariable( { kOutNames[i], "Output value at this control point" },
                    kDefaults[i],
                    [i]( Curve* p, float f ) { p->mPointOut[i] = f; } );
            }

            description =
                "Piecewise-linear response curve through up to 8 control points\n"
                "Author points with ascending In values - the node does not sort\n"
                "Below the first point the first Out is returned, past the last\n"
                "point the last Out";
        }
    };
#endif

    /** @brief CellularValue reduced to its cheapest configuration.
     *
     *  Closest cell only (Value Index 0), EuclideanSquared distance, no Size
     *  Jitter, no Minkowski. Bit-identical to a CellularValue left at exactly
     *  those defaults, at a fraction of the per-cell work: the slot insertion
     *  loop collapses to one compare+select and the distance switch is gone.
     *  Use the full CellularValue when any of the fixed knobs must move.
     */
    class CellularValueSimple : public virtual VariableRange<Seeded<ScalableGenerator>>
    {
    public:
        const Metadata& GetMetadata() const override;

        void SetGridJitter( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mGridJitter, gen ); }
        void SetGridJitter( float value ) { mGridJitter = value; }

    protected:
        HybridSource mGridJitter = 1.0f;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<CellularValueSimple> : MetadataT<VariableRange<Seeded<ScalableGenerator>>>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT()
        {
            groups.push_back( "Coherent Noise" );
            this->AddHybridSource( { "Grid Jitter", "How much to displace cells from their uniform grid position\n"
                                                    "0.0 will output a uniform grid\n"
                                                    "Above 1.0 will cause grid artifacts" },
                1.0f, &CellularValueSimple::SetGridJitter, &CellularValueSimple::SetGridJitter );

            description =
                "CellularValue reduced to closest-cell / EuclideanSquared / no jitter extras\n"
                "Bit-identical to CellularValue at those exact defaults, noticeably cheaper\n"
                "Reach for the full node when you need any of the removed knobs";
        }
    };
#endif

    /** @brief CellularDistance reduced to its cheapest configuration.
     *
     *  Closest distance only (Index0), EuclideanSquared, no Size Jitter, no
     *  Minkowski. Bit-identical to a CellularDistance left at exactly those
     *  defaults: the four-slot distance bookkeeping collapses to a single Min
     *  per cell and the return-type switch is gone.
     */
    class CellularDistanceSimple : public virtual VariableRange<Seeded<ScalableGenerator>>
    {
    public:
        const Metadata& GetMetadata() const override;

        void SetGridJitter( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mGridJitter, gen ); }
        void SetGridJitter( float value ) { mGridJitter = value; }

    protected:
        HybridSource mGridJitter = 1.0f;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<CellularDistanceSimple> : MetadataT<VariableRange<Seeded<ScalableGenerator>>>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT()
        {
            groups.push_back( "Coherent Noise" );
            this->AddHybridSource( { "Grid Jitter", "How much to displace cells from their uniform grid position\n"
                                                    "0.0 will output a uniform grid\n"
                                                    "Above 1.0 will cause grid artifacts" },
                1.0f, &CellularDistanceSimple::SetGridJitter, &CellularDistanceSimple::SetGridJitter );

            description =
                "CellularDistance reduced to Index0 / EuclideanSquared / no jitter extras\n"
                "Bit-identical to CellularDistance at those exact defaults, noticeably cheaper\n"
                "Reach for the full node when you need any of the removed knobs";
        }
    };
#endif

    /** @brief How BoxDomain evaluates its planes. */
    enum class BoxDomainMode
    {
        Strict, ///< Reproduces the Min(Gradient, ...) chain bit-for-bit, dead-axis terms included.
        Fast,   ///< Skips dead-axis arithmetic. Not bit-identical to the chain (signed zeros), but cheaper.
    };

    constexpr static const char* kBoxDomainMode_Strings[] =
    {
        "Strict",
        "Fast",
    };

    /** @brief Fused axis-aligned domain: a from-to band per axis in one node.
     *
     *  Replaces the corpus idiom of Min-folding one Gradient half-space per
     *  bound (4 nodes for a prism, 6 for a box, plus the Min chain). Each axis
     *  has a Lo and a Hi plane in exact Gradient parametrisation
     *  `(pos + offset) * multiplier`; a plane with multiplier 0 is off. The
     *  active planes are Min-folded in fixed order: X Lo, X Hi, Y Lo, Y Hi,
     *  Z Lo, Z Hi, W Lo, W Hi. With no active plane the output is 0.
     *
     *  Sign convention matches the Domain slot: author Lo multipliers positive
     *  and Hi multipliers negative to get "inside is positive".
     *
     *  Smoothness 0 folds with a hard Min. Any other value (or a node input)
     *  folds with the MinSmooth maths instead, rounding the box edges; that
     *  path is not bit-comparable to a hard Min chain and is not meant to be.
     */
    class BoxDomain : public virtual Generator
    {
    public:
        const Metadata& GetMetadata() const override;

        template<Dim D>
        void SetLoOffset( float value ) { mLoOffset[(int)D] = value; }
        template<Dim D>
        void SetLoOffset( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mLoOffset[(int)D], gen ); }
        template<Dim D>
        void SetLoMultiplier( float value ) { mLoMultiplier[(int)D] = value; }

        template<Dim D>
        void SetHiOffset( float value ) { mHiOffset[(int)D] = value; }
        template<Dim D>
        void SetHiOffset( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mHiOffset[(int)D], gen ); }
        template<Dim D>
        void SetHiMultiplier( float value ) { mHiMultiplier[(int)D] = value; }

        void SetSmoothness( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mSmoothness, gen ); }
        void SetSmoothness( float value ) { mSmoothness = value; }
        void SetMode( BoxDomainMode value ) { mMode = value; }

    protected:
        PerDimensionVariable<HybridSource> mLoOffset = 0.0f;
        PerDimensionVariable<float> mLoMultiplier = 0.0f;
        PerDimensionVariable<HybridSource> mHiOffset = 0.0f;
        PerDimensionVariable<float> mHiMultiplier = 0.0f;
        HybridSource mSmoothness = 0.0f;
        BoxDomainMode mMode = BoxDomainMode::Strict;

        template<typename T>
        friend struct MetadataT;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<BoxDomain> : MetadataT<Generator>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT()
        {
            groups.push_back( "Basic Generators" );
            this->AddPerDimensionHybridSource( { "Lo Offset", "Offset of this axis' lower plane: (pos + offset) * multiplier" },
                0.0f, []( BoxDomain* p ) { return std::ref( p->mLoOffset ); } );
            this->AddPerDimensionVariable( { "Lo Multiplier", "Steepness of this axis' lower plane, 0 = plane off\nPositive means inside-is-positive for a lower bound" },
                0.0f, []( BoxDomain* p ) { return std::ref( p->mLoMultiplier ); }, 0.f, 0.f, 0.001f );
            this->AddPerDimensionHybridSource( { "Hi Offset", "Offset of this axis' upper plane: (pos + offset) * multiplier" },
                0.0f, []( BoxDomain* p ) { return std::ref( p->mHiOffset ); } );
            this->AddPerDimensionVariable( { "Hi Multiplier", "Steepness of this axis' upper plane, 0 = plane off\nNegative means inside-is-positive for an upper bound" },
                0.0f, []( BoxDomain* p ) { return std::ref( p->mHiMultiplier ); }, 0.f, 0.f, 0.001f );
            this->AddHybridSource( { "Smoothness",
                "0 = hard Min fold, bit-identical to the Gradient/Min chain\n"
                "Anything else rounds the box edges with the MinSmooth maths" },
                0.0f, &BoxDomain::SetSmoothness, &BoxDomain::SetSmoothness );
            this->AddVariableEnum( { "Mode",
                "Strict reproduces the Gradient chain bit-for-bit\n"
                "Fast skips dead-axis arithmetic (cheaper, not bit-identical)" },
                BoxDomainMode::Strict, &BoxDomain::SetMode, kBoxDomainMode_Strings );

            description =
                "An axis-aligned domain volume in a single node\n"
                "Each axis carries a lower and an upper plane in Gradient form\n"
                "(pos + offset) * multiplier, multiplier 0 = plane off\n"
                "Active planes are Min-folded: inside is positive when Lo\n"
                "multipliers are positive and Hi multipliers negative\n"
                "Replaces the Min chain of Gradient half-spaces";
        }
    };
#endif
}
