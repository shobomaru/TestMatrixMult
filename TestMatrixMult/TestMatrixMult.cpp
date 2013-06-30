// TestMatrixMult.cpp
//  4x4 Matrix Multiplication using SSE / AVX
//  written by @shobomaru


//----- Configure -----

// 
//#define PROFIACA

//----- Configure End -----


#include <iostream>
#include <cassert>

#define __AVX__
#include <immintrin.h>

#ifdef _MSC_VER
#   define ALIGN32 __declspec( align( 32 ) )
#elif
#   define ALIGN32 __attribute__(( aligned( 32 ) ))
#endif

#ifdef PROFIACA
#   include <iacaMarks.h>
#   define NOINLINE __attribute(( noinline ))
#else
#   define IACA_START
#   define IACA_END
#   define NOINLINE
#endif

void checkCPU() {
    
    // see Wikipedia "CPUID"
    
    unsigned int index = 1;
    unsigned int regs[ 4 ];
    
#ifdef _MSC_VER
    
    __cpuid( reinterpret_cast< int* >( regs ), index );
    
#elif
    
    __asm__ __volatile__(
#if defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)
                         "pushq %%rbx     \n\t" /* save %rbx */
#else
                         "pushl %%ebx     \n\t" /* save %ebx */
#endif
                         "cpuid            \n\t"
                         "movl %%ebx ,%[ebx]  \n\t" /* write the result into output var */
#if defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)
                         "popq %%rbx \n\t"
#else
                         "popl %%ebx \n\t"
#endif
                         : "=a"(regs[0]), [ebx] "=r"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
                         : "a"(index));
    
#endif
    
    if( regs[ 3 ] & 0x02000000 ) puts( "support SSE" );
    if( regs[ 3 ] & 0x04000000 ) puts( "support SSE2" );
    if( regs[ 2 ] & 0x00000001 ) puts( "support SSE3" );
    if( regs[ 2 ] & 0x00080000 ) puts( "support SSE4.1" );
    if( regs[ 2 ] & 0x00100000 ) puts( "support SSE4.2" );
    if( regs[ 2 ] & 0x10000000 ) puts( "support AVX" );
    
    index = 7;
    
#ifdef _MSC_VER
    
    __cpuidex( reinterpret_cast< int* >( regs ), index, 0 );
    
#elif
    
    __asm__ __volatile__(
#if defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)
                         "pushq %%rbx     \n\t" /* save %rbx */
#else
                         "pushl %%ebx     \n\t" /* save %ebx */
#endif
                         "cpuid            \n\t"
                         "movl %%ebx ,%[ebx]  \n\t" /* write the result into output var */
#if defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)
                         "popq %%rbx \n\t"
#else
                         "popl %%ebx \n\t"
#endif
                         : "=a"(regs[0]), [ebx] "=r"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
                         : "a"(index));
    
#endif
    
    if( regs[ 1 ] & 0x00000020 ) puts( "support AVX2" );
    
    
    puts("");
}

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
#elif
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
#elif
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

static void NOINLINE mul32x4( const __m128 *v1, const __m128 *v2, __m128 *vout )
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

static void mul32x8( const __m256 *v1, const __m256 *v2, __m256 *vout )
{
    for( int r = 0; r < 4; r++ ) {
    }
}

static void NOINLINE inv( const float *v, float *vout )
{
}

int main(int argc, const char * argv[])
{
    checkCPU();
    
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
    
    std::cout << "Scalar Mult" << std::endl;
    mul( a1, a2, aout );
    trace( aout, 4 );
    
    std::cout << "SSE Mult" << std::endl;
    mul32x4( x1, x2, xout );
    trace( xout, 4 );
    
    std::cout << "AVX Mult" << std::endl;
    mul32x8( y1, y2, yout );
    trace( yout, 4 );
    
    return 0;
}

