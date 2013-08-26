// TestMatrixMult.cpp
//  4x4 Matrix Manipulation using SSE1 / AVX2



/* ----- LICENSE (The zlib/libpng License) ----- */

// Copyright (c) 2013 shobomaru
// 
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
// 
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 
//    1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 
//    2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 
//    3. This notice may not be removed or altered from any source
//    distribution.



//----- Configure -----

// Enable IACA profiling
// Please change the project path before build.
//#define PROFIACA

//----- Configure End -----


#include <iostream>
#include <cassert>

//#define __AVX__
#include <immintrin.h>

#ifdef _MSC_VER
#   define ALIGN32 __declspec( align( 32 ) )
#else
#   define ALIGN32 __attribute__(( aligned( 32 ) ))
#endif

#ifdef PROFIACA
#   include <iacaMarks.h>
#   ifdef _MSC_VER
#       define NOINLINE __declspec( noinline )
#   else
#       define NOINLINE __attribute(( noinline ))
#   endif
#   pragma comment( lib, "iacaLoader.lib" )
#else
#   define IACA_START
#   define IACA_END
#   define NOINLINE
#endif

static void trace( const float *v, int numr )
{
    for( int r = 0; r < numr; r++ ) {
        std::cout << "( " << v[ r * 4 + 0 ]
                << ", " << v[ r * 4 + 1 ]
                << ", " << v[ r * 4 + 2 ]
                << ", " << v[ r * 4 + 3 ] << " )" << std::endl;
    }
}

static void trace( const __m128 *v, int numr )
{
    for( int r = 0; r < numr; r++ ) {
#ifdef _MSC_VER
        std::cout << "( " << v[ r ].m128_f32[ 0 ]
                << ", " << v[ r ].m128_f32[ 1 ]
                << ", " << v[ r ].m128_f32[ 2 ]
                << ", " << v[ r ].m128_f32[ 3 ] << " )" << std::endl;
#else
        std::cout << "( " << v[ r ][ 0 ]
                << ", " << v[ r ][ 1 ]
                << ", " << v[ r ][ 2 ]
                << ", " << v[ r ][ 3 ] << " )" << std::endl;
#endif
    }
}

static void trace( const __m256 *v, int numr )
{
    assert( numr % 2 == 0 );
    for( int r = 0; r < numr / 2; r++ ) {
#ifdef _MSC_VER
        std::cout << "( " << v[ r ].m256_f32[ 0 ]
                << ", " << v[ r ].m256_f32[ 1 ]
                << ", " << v[ r ].m256_f32[ 2 ]
                << ", " << v[ r ].m256_f32[ 3 ] << " )" << std::endl
                << "( " << v[ r ].m256_f32[ 4 ]
                << ", " << v[ r ].m256_f32[ 5 ]
                << ", " << v[ r ].m256_f32[ 6 ]
                << ", " << v[ r ].m256_f32[ 7 ] << " )" << std::endl;
#else
        std::cout << "( " << v[ r ][ 0 ]
                << ", " << v[ r ][ 1 ]
                << ", " << v[ r ][ 2 ]
                << ", " << v[ r ][ 3 ] << " )" << std::endl
                << "( " << v[ r ][ 4 ]
                << ", " << v[ r ][ 5 ]
                << ", " << v[ r ][ 6 ]
                << ", " << v[ r ][ 7 ] << " )" << std::endl;
#endif
    }
}

static void NOINLINE mul( const float *v1, const float *v2, float *vout )
{
    for( int r = 0; r < 4; r++ ) {
        for( int c = 0; c < 4; c++ ) {
            float a = v1[ r * 4 + 0 ] * v2[ 0 * 4 + c ];
            a += v1[ r * 4 + 1 ] * v2[ 1 * 4 + c ];
            a += v1[ r * 4 + 2 ] * v2[ 2 * 4 + c ];
            a += v1[ r * 4 + 3 ] * v2[ 3 * 4 + c ];
            vout[ r * 4 + c ] = a;
        }
    }
}

static void NOINLINE mulX4( const __m128 *v1, const __m128 *v2, __m128 *vout )
{
    for( int r = 0; r < 4; r++ ) {
        __m128 a0 = _mm_shuffle_ps( v1[ r ], v1[ r ], _MM_SHUFFLE( 0, 0, 0, 0 ) );
        __m128 a1 = _mm_shuffle_ps( v1[ r ], v1[ r ], _MM_SHUFFLE( 1, 1, 1, 1 ) );
        __m128 a2 = _mm_shuffle_ps( v1[ r ], v1[ r ], _MM_SHUFFLE( 2, 2, 2, 2 ) );
        __m128 a3 = _mm_shuffle_ps( v1[ r ], v1[ r ], _MM_SHUFFLE( 3, 3, 3, 3 ) );
        
        __m128 b0 = _mm_mul_ps( a0, v2[ 0 ] );
        __m128 b1 = _mm_mul_ps( a1, v2[ 1 ] );
        __m128 b2 = _mm_mul_ps( a2, v2[ 2 ] );
        __m128 b3 = _mm_mul_ps( a3, v2[ 3 ] );
        
        __m128 c0 = _mm_add_ps( b0, b1 );
        __m128 c1 = _mm_add_ps( b2, b3 );
        vout[ r ] = _mm_add_ps( c0, c1 );
    }
}

static void NOINLINE mulX8( const __m256 *v1, const __m256 *v2, __m256 *vout )
{
    static const int ALIGN32 p1[ 8 ] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    static const int ALIGN32 p2[ 8 ] = { 2, 2, 2, 2, 3, 3, 3, 3 };
    static const int ALIGN32 p3[ 8 ] = { 4, 4, 4, 4, 5, 5, 5, 5 };
    static const int ALIGN32 p4[ 8 ] = { 6, 6, 6, 6, 7, 7, 7, 7 };
    const __m256i perm1 = _mm256_load_si256( reinterpret_cast< const __m256i* >( p1 ) );
    const __m256i perm2 = _mm256_load_si256( reinterpret_cast< const __m256i* >( p2 ) );
    const __m256i perm3 = _mm256_load_si256( reinterpret_cast< const __m256i* >( p3 ) );
    const __m256i perm4 = _mm256_load_si256( reinterpret_cast< const __m256i* >( p4 ) );
    for( int r = 0; r < 2; r++ ) {
        __m256 a0 = _mm256_permutevar8x32_ps( v1[ r ], perm1 );
        __m256 a1 = _mm256_permutevar8x32_ps( v1[ r ], perm2 );
        __m256 a2 = _mm256_permutevar8x32_ps( v1[ r ], perm3 );
        __m256 a3 = _mm256_permutevar8x32_ps( v1[ r ], perm4 );
        
        __m256 b0 = _mm256_mul_ps( a0, v2[ 0 ] );
        __m256 b1 = _mm256_mul_ps( a1, v2[ 1 ] );
        __m256 b2 = _mm256_mul_ps( a2, v2[ 0 ] );
        __m256 b3 = _mm256_mul_ps( a3, v2[ 1 ] );
        
        __m256 c0 = _mm256_add_ps( b0, b1 );
        __m256 c1 = _mm256_add_ps( b2, b3 );
        __m256 d0 = _mm256_permute2f128_ps( c0, c1, _MM_SHUFFLE( 0, 2, 0, 0 ) );
        __m256 d1 = _mm256_permute2f128_ps( c0, c1, _MM_SHUFFLE( 0, 3, 0, 1 ) );
        vout[ r ] = _mm256_add_ps( d0, d1 );
    }
}

static void NOINLINE transpose( const float *v1, float *vout )
{
    for( int r = 0; r < 4; r++ ) {
        for( int c = 0; c < 4; c++ ) {
            vout[ c * 4 + r ] = v1[ r * 4 + c ];
        }
    }
}

static void NOINLINE transposeX4( const __m128 *v1, __m128 *vout )
{
    __m128 a0 = _mm_unpacklo_ps( v1[ 0 ], v1[ 2 ] );
    __m128 a1 = _mm_unpacklo_ps( v1[ 1 ], v1[ 3 ] );
    __m128 a2 = _mm_unpackhi_ps( v1[ 0 ], v1[ 2 ] );
    __m128 a3 = _mm_unpackhi_ps( v1[ 1 ], v1[ 3 ] );
    vout[ 0 ] = _mm_unpacklo_ps( a0, a1 );
    vout[ 1 ] = _mm_unpackhi_ps( a0, a1 );
    vout[ 2 ] = _mm_unpacklo_ps( a2, a3 );
    vout[ 3 ] = _mm_unpackhi_ps( a2, a3 );
}

static void NOINLINE transposeX8( const __m256 *v1, __m256 *vout )
{
#if 0 // AVX1
    __m256 a0 = _mm256_unpacklo_ps( v1[ 0 ], v1[ 1 ] );
    __m256 a1 = _mm256_unpackhi_ps( v1[ 0 ], v1[ 1 ] );
    __m256 b0 = _mm256_permute2f128_ps( a0, a1, _MM_SHUFFLE( 0, 2, 0, 0 ) );
    __m256 b1 = _mm256_permute2f128_ps( a0, a1, _MM_SHUFFLE( 0, 3, 0, 1 ) );
    __m256 c0 = _mm256_unpacklo_ps( b0, b1 );
    __m256 c1 = _mm256_unpackhi_ps( b0, b1 );
    vout[ 0 ] = _mm256_permute2f128_ps( c0, c1, _MM_SHUFFLE( 0, 2, 0, 0 ) );
    vout[ 1 ] = _mm256_permute2f128_ps( c0, c1, _MM_SHUFFLE( 0, 3, 0, 1 ) );
#else // AVX2
    static const int ALIGN32 p1[ 8 ] = { 0, 4, 2, 6, 1, 5, 3, 7 };
    static const int ALIGN32 p2[ 8 ] = { 2, 6, 0, 4, 3, 7, 1, 5 };
    const __m256i perm1 = _mm256_load_si256( reinterpret_cast< const __m256i* >( p1 ) );
    const __m256i perm2 = _mm256_load_si256( reinterpret_cast< const __m256i* >( p2 ) );
    __m256 a0 = _mm256_permutevar8x32_ps( v1[ 0 ], perm1 );
    __m256 a1 = _mm256_permutevar8x32_ps( v1[ 1 ], perm2 );
    vout[ 0 ] = _mm256_blend_ps( a0, a1, 0xCC );
    vout[ 1 ] = _mm256_shuffle_ps( a0, a1, 0x4E );
#endif
}

int main(int argc, const char * argv[])
{
    ALIGN32 float a1[ 16 ] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };
    ALIGN32 float a2[ 16 ] = {
        15, 12, 4, 7, 9, 0, 3, 13, 6, 10, 1, 8, 5, 11, 2, 14
    };
    ALIGN32 float aout[ 16 ];
    
    __m128 x1[ 4 ] = { _mm_load_ps( a1 ), _mm_load_ps( a1 + 4 ), _mm_load_ps( a1 + 8 ), _mm_load_ps( a1 + 12 ) };
    __m128 x2[ 4 ] = { _mm_load_ps( a2 ), _mm_load_ps( a2 + 4 ), _mm_load_ps( a2 + 8 ), _mm_load_ps( a2 + 12 ) };
    __m128 xout[ 4 ];
    
    __m256 y1[ 2 ] = { _mm256_load_ps( a1 ), _mm256_load_ps( a1 + 8 ) };
    __m256 y2[ 2 ] = { _mm256_load_ps( a2 ), _mm256_load_ps( a2 + 8 ) };
    __m256 yout[ 2 ];
    
    std::cout << "FPU Mult" << std::endl;
    mul( a1, a2, aout );
    trace( aout, 4 );
    
    std::cout << "SSE Mult" << std::endl;
    mulX4( x1, x2, xout );
    trace( xout, 4 );
    
    std::cout << "AVX2 Mult" << std::endl;
    mulX8( y1, y2, yout );
    trace( yout, 4 );
    
    std::cout << "FPU Transpose" << std::endl;
    transpose( a1, aout );
    trace( aout, 4 );
    
    std::cout << "SSE Transpose" << std::endl;
    transposeX4( x1, xout );
    trace( xout, 4 );
    
    std::cout << "AVX Transpose" << std::endl;
    transposeX8( y1, yout );
    trace( yout, 4 );
    
    return 0;
}
