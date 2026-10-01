# Tutorial: a² − ab + b² in C++ with Wrap and Clamp overflow

Sep 30, 2026 · @hojun lee

## Overview

We build one C++ header, `quad_form.hpp`, that computes f(a, b) = a² − ab + b² for every type below with a chosen overflow policy, plus a fast function that processes whole images. The integer part rests on one observation: the exact answer always fits in twice the input width, so we compute it exactly and then wrap or clamp.

| Family | Widths | Stored in |
| --- | --- | --- |
| Unsigned integer | 1, 2, 3, 4, 8, 9–16, 32, 64 bits | `uint8_t` / `uint16_t` / `uint32_t` / `uint64_t` |
| Signed integer | 8, 9–16, 32, 64 bits | `int8_t` / `int16_t` / `int32_t` / `int64_t` |
| Floating point | 32, 64 bits | `float`, `double` |

The two overflow policies:

- **Wrap** keeps only the low N bits of the true result, like hardware unsigned arithmetic. For signed types those N bits are read as two's complement.
- **Clamp** (saturate) returns the largest representable value when the true result is too big.

Everything is a plain function: the bit width, signedness and policy are ordinary arguments. The finished API:

```cpp
// One value at a time
quad_unsigned(300, 100, 9, Overflow::Clamp);   // 9-bit unsigned  -> 511
quad_signed(-128, 127, 8, Overflow::Wrap);     // 8-bit signed    -> -127
quad_float(1.5f, 2.5f, Overflow::Clamp);
quad_double(1e200, 2e200, Overflow::Wrap);

// Whole images (use this for bulk data)
quad_image(a, b, out, pixel_count, 12, Overflow::Clamp);   // 12-bit pixels in uint16_t
quad_image(fa, fb, fout, pixel_count, Overflow::Wrap);     // float image
```

On 10 million pixels built with `-O3 -march=native`, `quad_image` takes 1–3 ms for 8- to 16-bit integer images and 13–18 ms for float images (Step 5 has the measurements).

You need a C++17 compiler (C++20 recommended). The header uses only standard integer types up to 64 bits, with no 128-bit integers, so it works with any standard C++17 compiler, including MSVC. The test program needs GCC or Clang (see Reproducing).

## The math that makes it easy

Three facts about the formula decide the whole design: the result is never negative, it fits in 2N bits, and unsigned C++ arithmetic is exact modulo a power of two.

### Fact 1: the result is never negative

Complete the square:

```latex
a^2 - ab + b^2 = \left(a - \tfrac{b}{2}\right)^2 + \tfrac{3}{4}b^2 \ge 0
```

Both terms are squares, so the sum is at least zero for every real a and b. Consequence: **clamping only needs an upper limit.** We never clamp to MIN, even for signed types.

### Fact 2: the result fits in 2N bits

For unsigned N-bit inputs, let m = max(a, b) and k = min(a, b). Then f = m² + k(k − m), and k(k − m) ≤ 0, so:

```latex
0 \le f(a, b) \le \max(a, b)^2 \le (2^N - 1)^2 < 2^{2N}
```

For signed N-bit inputs the worst case is a = MIN, b = MAX (opposite signs make −ab positive). With MIN = −2^(N−1) and MAX = 2^(N−1) − 1:

```latex
f(\text{MIN}, \text{MAX}) < 3 \cdot 2^{2N-2} < 2^{2N}
```

So in both cases the exact answer is a non-negative number below 2^(2N).

### Fact 3: unsigned arithmetic is exact modulo 2^W

In C++, arithmetic on a W-bit unsigned type is defined as arithmetic modulo 2^W. Converting a negative number to that type is also modulo 2^W (−1 becomes 2^W − 1). So if we:

1. convert a and b to an unsigned type with W ≥ 2N bits,
2. compute `x*x - x*y + y*y` there,

the answer we get is the true answer modulo 2^W. By Fact 2 the true answer is already in \[0, 2^W), so reducing it modulo 2^W changes nothing. **The unsigned result is the exact result**, even when intermediate steps "wrap around" and even for negative inputs.

Fact 3 needs W ≥ 2N for an *exact* result, but the two policies need different things, and that is how the code avoids any type wider than 64 bits:

- **Wrap only needs the low N bits.** They are the same whether we compute modulo 2^W or exactly, as long as W ≥ N. So `uint64_t` (W = 64) handles Wrap for every width up to 64.
- **Clamp needs the exact result** to compare it with MAX. With `uint64_t` that works whenever 2N ≤ 64, which means widths up to 32 bits.
- **Clamp for 33 to 64 bits** would need more than 64 bits. Instead of a 128-bit type, the code uses checked arithmetic on the magnitudes of a and b (Step 2), which never needs anything wider than `uint64_t`.

| Case | How the result is computed |
| --- | --- |
| Wrap, any width | `uint64_t`, modulo 2^64 (low N bits are exact) |
| Clamp, 1–32 bits | `uint64_t`, exact (64 ≥ 2 × 32) |
| Clamp, 33–64 bits | checked arithmetic in `uint64_t` |
| `quad_image`, pixels of 8 or 16 bits | `uint32_t`, exact (32 ≥ 2 × 16) |

With the exact answer in hand, clamping means "compare with MAX"; wrapping means "keep the low N bits". Floats need a different argument, covered in Step 4.

## Step 1: Helper functions for N-bit ranges

Five small `constexpr` functions answer every question the algorithms ask about an N-bit type: which bits are valid, what its range is, and how to read the low bits as a signed number.

Start the header with the includes and the policy enum. Only standard headers are needed:

```cpp
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>

enum class Overflow { Wrap, Clamp };
```

Then the helpers:

```cpp
// Mask with the low `bits` bits set, e.g. 0x1FF for 9 bits.
// (Shifting right avoids 1 << 64, which is undefined.)
constexpr std::uint64_t low_bits_mask(unsigned bits) {
    return ~std::uint64_t{0} >> (64 - bits);
}

// Range of an N-bit integer.
constexpr std::uint64_t max_unsigned(unsigned bits) { return low_bits_mask(bits); }
constexpr std::int64_t  max_signed(unsigned bits)   { return static_cast<std::int64_t>(low_bits_mask(bits) >> 1); }
constexpr std::int64_t  min_signed(unsigned bits)   { return -max_signed(bits) - 1; }

// Treats the low `bits` bits of `value` as a two's-complement number and
// copies its sign bit into all higher bits.
constexpr std::uint64_t sign_extend(std::uint64_t value, unsigned bits) {
    const std::uint64_t sign_bit = std::uint64_t{1} << (bits - 1);

    const bool is_negative = (value & sign_bit) != 0;
    if (is_negative) {
        value |= ~low_bits_mask(bits);    // fill the upper bits with 1s
    }
    return value;
}
```

What each one does:

- **`low_bits_mask`** has the low N bits set: `0x1FF` for 9 bits, all ones for 64. It shifts all-ones *right* by `64 - bits`. The obvious `(1 << bits) - 1` is undefined for 64 bits, because shifting by the full width is not allowed.
- **`max_unsigned`** is 2^N − 1, which is just the mask.
- **`max_signed`** drops one more bit from the mask: for 8 bits, `0xFF >> 1 = 127`. **`min_signed`** is −max − 1, so −128.
- **`sign_extend`** fills everything above bit N−1 with copies of bit N−1. For 8 bits, `0x81` (129) becomes `0xFFFF'FFFF'FFFF'FF81`, which is −127 as a 64-bit number. For 64 bits, `~low_bits_mask` is 0 and nothing changes, which is correct.

Because they are `constexpr`, you can check them at compile time: `static_assert(max_signed(12) == 2047 && min_signed(12) == -2048);`.

The bit width is an ordinary argument rather than a template parameter, so one compiled function handles every width. That matters when the width is only known at run time, for example read from an image file's header.

**One rule for odd widths:** a 9-bit value stored in a `uint16_t` must actually be in \[0, 511\]. The single-value functions check this with `assert`; `quad_image` does not check each pixel, for speed.

## Step 2: The arithmetic building blocks

Four small functions do all the integer arithmetic. `quad_mod64` gives the result modulo 2^64, which covers Wrap and Clamp up to 32 bits; `add_product`, `clamped_quad` and `magnitude` give an exact clamped result for 33 to 64 bits without any type wider than 64 bits.

### 2a. The result modulo 2^64

```cpp
// a*a - a*b + b*b modulo 2^64.
//
// Negative inputs are passed in converted to uint64_t (-1 becomes 2^64 - 1).
// Unsigned arithmetic is defined as modulo 2^64, so the low bits of this
// result are always the low bits of the true result. When the true result
// is below 2^64 (inputs of up to 32 bits), this is the exact answer.
constexpr std::uint64_t quad_mod64(std::uint64_t x, std::uint64_t y) {
    return x * x - x * y + y * y;
}
```

This is Fact 3 with W = 64. Callers convert their inputs as they pass them in:

- **Unsigned:** `quad_mod64(a, b)` with `uint64_t` arguments.
- **Signed:** `quad_mod64(static_cast<std::uint64_t>(a), static_cast<std::uint64_t>(b))`. Converting a negative number to an unsigned type is defined as modulo 2^64, which is exactly sign extension: −1 becomes 2^64 − 1.

Intermediate values may "wrap around" (for example `x * y` when one input is negative), but the final value is still correct modulo 2^64. The Worked examples section traces this step by step. All arithmetic happens in an unsigned type, where overflow is defined behaviour; doing the same in `int64_t` would be undefined as soon as a product overflowed.

### 2b. Clamping 33- to 64-bit values without a wider type

For widths above 32 bits the exact result can exceed 2^64, so `quad_mod64` cannot tell whether it is above MAX. The fix is to rearrange the formula so every term is non-negative, then add the terms one at a time while checking each step against MAX.

Let big = max(|a|, |b|) and small = min(|a|, |b|). Flipping the signs of both a and b does not change the result, and with opposite signs −ab = |a||b|, so:

```latex
\text{same sign: } f = \text{big}\,(\text{big} - \text{small}) + \text{small}^2 \qquad \text{opposite sign: } f = \text{big}^2 + \text{big}\cdot\text{small} + \text{small}^2
```

Every term is ≥ 0 and at most f, so a running total only grows. As soon as it would pass MAX, the answer is MAX; if it never does, the total is the exact answer. Each step uses this helper:

```cpp
// Adds x * y to `total` if the new total stays at most `limit`, and returns
// true. Returns false, leaving `total` unchanged, if it would exceed `limit`.
// Nothing here can overflow: the division tests x * y <= limit without
// computing x * y first.
constexpr bool add_product(std::uint64_t& total, std::uint64_t x, std::uint64_t y,
                           std::uint64_t limit) {
    if (x != 0 && y > limit / x) {
        return false;                       // x * y alone exceeds limit
    }
    const std::uint64_t product = x * y;
    if (product > limit - total) {
        return false;                       // total + product exceeds limit
    }
    total += product;
    return true;
}
```

- **The product test is exact.** For integers, x · y > L exactly when y > ⌊L / x⌋, so the division decides the question without ever forming the too-large product.
- **The sum test cannot overflow either.** `total` is always ≤ `limit`, so `limit - total` is a valid unsigned value, and `product > limit - total` is the same as `total + product > limit`.

The clamp itself:

```cpp
// min(a*a - a*b + b*b, max), computed from |a| and |b| without overflow.
constexpr std::uint64_t clamped_quad(std::uint64_t abs_a, std::uint64_t abs_b,
                                     bool same_sign, std::uint64_t max) {
    // Quick exit, no division: the result is never below (3/4) * big^2 (its
    // smallest value, reached at small = big / 2). Once big >= 2^33 that is
    // more than 2^64, so the result is certainly above max. (abs_a | abs_b)
    // has a bit at position 33 or higher exactly when big >= 2^33.
    if (((abs_a | abs_b) >> 33) != 0) {
        return max;
    }

    const std::uint64_t big   = abs_a > abs_b ? abs_a : abs_b;
    const std::uint64_t small = abs_a > abs_b ? abs_b : abs_a;

    std::uint64_t total = 0;
    if (same_sign) {
        if (!add_product(total, big, big - small, max)) return max;
    } else {
        if (!add_product(total, big, big,   max)) return max;
        if (!add_product(total, big, small, max)) return max;
    }
    if (!add_product(total, small, small, max)) return max;
    return total;
}
```

The quick exit matters for speed. Large values are common in 64-bit data, and for them the answer is obviously MAX: f is smallest when small = big / 2, where it equals ¾ · big², so once big ≥ 2^33 the result is at least ¾ · 2^66 = 3 · 2^64, far above any MAX. The test is written as `(abs_a | abs_b) >> 33` rather than comparing `big` because the OR has no data-dependent branch. In testing, an earlier version that first computed `big` with a comparison spent much of its time on branch mispredictions with random data: clamping 10 million random 64-bit signed pixels took 61 ms with it and 21 ms with the OR.

### 2c. Magnitudes

```cpp
// |a| as an unsigned number. Works for INT64_MIN, whose magnitude 2^63 does
// not fit in int64_t (negating it as int64_t would be undefined).
constexpr std::uint64_t magnitude(std::int64_t a) {
    const std::uint64_t bits_of_a = static_cast<std::uint64_t>(a);
    return a < 0 ? 0 - bits_of_a : bits_of_a;
}
```

`std::abs(INT64_MIN)` is undefined because +2^63 does not fit in `int64_t`. Negating in unsigned arithmetic instead (0 − x modulo 2^64) gives exactly 2^63.

## Step 3: Wrap and clamp for integers

Each integer function checks its arguments, then picks one of three paths: Wrap uses `quad_mod64` for any width; Clamp uses `quad_mod64` exactly for widths up to 32 bits; and Clamp for 33–64 bits uses `clamped_quad`. Clamp only needs an upper limit, because the result is never negative (Fact 1).

### Unsigned

```cpp
// a and b must be N-bit unsigned values: 0 <= a, b <= 2^bits - 1.
inline std::uint64_t quad_unsigned(std::uint64_t a, std::uint64_t b,
                                   unsigned bits, Overflow policy) {
    assert(bits >= 1 && bits <= 64);
    assert(a <= max_unsigned(bits) && b <= max_unsigned(bits));

    if (policy == Overflow::Wrap) {
        return quad_mod64(a, b) & low_bits_mask(bits);     // low `bits` bits
    }

    const std::uint64_t max = max_unsigned(bits);
    if (bits <= 32) {
        // The exact result is below 2^64, so quad_mod64 is exact: fast path.
        const std::uint64_t exact = quad_mod64(a, b);
        return exact > max ? max : exact;
    }
    return clamped_quad(a, b, /*same_sign=*/true, max);
}
```

- **Wrap:** the mask keeps the low N bits of the mod-2^64 result, which is the result modulo 2^N.
- **Clamp up to 32 bits:** the exact result is below 2^64 (Fact 2), so it is computed directly and compared with 2^N − 1.
- **Clamp above 32 bits:** unsigned values always have the same sign, so `clamped_quad` uses the same-sign arrangement.

### Signed

```cpp
// a and b must be N-bit signed values: -2^(bits-1) <= a, b <= 2^(bits-1) - 1.
inline std::int64_t quad_signed(std::int64_t a, std::int64_t b,
                                unsigned bits, Overflow policy) {
    assert(bits >= 2 && bits <= 64);
    assert(a >= min_signed(bits) && a <= max_signed(bits));
    assert(b >= min_signed(bits) && b <= max_signed(bits));

    // Converting a negative int64_t to uint64_t sign-extends it (modulo 2^64).
    const std::uint64_t x = static_cast<std::uint64_t>(a);
    const std::uint64_t y = static_cast<std::uint64_t>(b);

    if (policy == Overflow::Wrap) {
        const std::uint64_t low_bits = quad_mod64(x, y) & low_bits_mask(bits);
        return static_cast<std::int64_t>(sign_extend(low_bits, bits));
    }

    // The result is never negative, so clamping only needs an upper limit.
    const std::uint64_t max = static_cast<std::uint64_t>(max_signed(bits));
    if (bits <= 32) {
        // The exact result is below 3 * 2^62 < 2^64, so quad_mod64 is exact.
        const std::uint64_t exact = quad_mod64(x, y);
        return static_cast<std::int64_t>(exact > max ? max : exact);
    }
    const bool same_sign = (a < 0) == (b < 0);
    return static_cast<std::int64_t>(clamped_quad(magnitude(a), magnitude(b), same_sign, max));
}
```

- **Wrap:** after masking, a negative result sits in the low N bits as an unsigned pattern (for 8 bits, −127 appears as `0x81`). `sign_extend` turns it back into a proper 64-bit value, and the final cast to `int64_t` is the one place that relies on C++20's two's-complement guarantee (see Portability).
- **Clamp up to 32 bits:** the same comparison, against 2^(N−1) − 1. We never clamp to MIN.
- **Clamp above 32 bits:** `magnitude` gives |a| and |b| (safely, even for INT64\_MIN), and the sign test picks the arrangement in `clamped_quad`.

### Using the results

- **Argument checks:** the `assert`s catch a bad bit width or an input outside the N-bit range, such as 600 passed as a 9-bit value. Building with `-DNDEBUG` removes them.
- **Return types:** the functions always return `uint64_t` or `int64_t`. The value is always within the N-bit range, so storing it in the smaller storage type is a safe cast: `auto r = static_cast<std::uint16_t>(quad_unsigned(a, b, 9, Overflow::Clamp));`.
- **Speed:** about 1 ns per value for widths up to 32 bits, and about 2 ns for 64-bit values. For images, `quad_image` (Step 5) is faster still.

## Step 4: Float and double

Floats cannot wrap, so the two policies become: **Wrap** = IEEE default (overflow gives +infinity) and **Clamp** = overflow gives the largest finite value. The code uses the plain formula and falls back to a rearranged one only in the rare case where the plain result is not finite.

### The plain formula is accurate

Because (|a| − |b|)² ≥ 0, the cross term can never dominate:

```latex
|ab| \le \frac{a^2 + b^2}{2} \le a^2 - ab + b^2
```

So each of the three terms a², ab and b² is at most twice the final result. Their rounding errors are therefore small compared with the answer, and `a*a - a*b + b*b` is accurate to within a couple of units in the last place. There is no harmful cancellation.

### Its one weakness: intermediate overflow

The plain formula computes a² first. If a² alone exceeds the type's maximum, it becomes infinity even when the final answer is finite:

| Inputs (double) | True result | Plain formula |
| --- | --- | --- |
| a = 1.4e154, b = 0.7e154 | 1.47e308 (finite) | inf |
| a = b = 1.4e154 | 1.96e308 (overflow) | inf − inf + inf = NaN |

The first row is a wrong answer; the second should be +inf but is NaN.

### The fallback: make every term non-negative

Pick the arrangement by the signs of a and b:

```latex
\text{same sign: } f = (a - b)^2 + ab \qquad \text{opposite sign: } f = a^2 + b^2 + |ab|
```

Every term is ≥ 0 and at most the final result, so an intermediate can only overflow if the answer itself does. (With the same sign, |a − b| ≤ max(|a|, |b|), so a − b cannot overflow either.)

```cpp
inline double quad_all_terms_nonnegative(double a, double b) {
    const bool same_sign = (a >= 0) == (b >= 0);

    if (same_sign) {
        const double difference = a - b;
        return difference * difference + a * b;
    }
    return a * a + b * b - a * b;   // ab < 0, so this adds |ab|
}
```

### Combining them

```cpp
inline double quad_fp(double a, double b) {
    const double plain = a * a - a * b + b * b;
    if (std::isfinite(plain)) {
        return plain;                              // almost always
    }
    return quad_all_terms_nonnegative(a, b);       // rare: overflow or infinite input
}
```

A finite plain result means no intermediate overflowed, because infinity and NaN always propagate to the sum. So the plain answer can be trusted whenever it is finite.

Why not always use the rearranged form? Its branch on the signs of a and b is unpredictable on real image data, and the CPU keeps mispredicting it. On 10 million float pixels the rearranged form alone took about 78 ms; `quad_fp` takes about 23 ms. Its own branch is almost never taken, so the CPU predicts it correctly.

### Double and float

```cpp
inline double quad_double(double a, double b, Overflow policy) {
    const double result = quad_fp(a, b);

    const bool inputs_finite = std::isfinite(a) && std::isfinite(b);
    const bool overflowed    = inputs_finite && std::isinf(result);

    if (policy == Overflow::Clamp && overflowed) {
        return std::numeric_limits<double>::max();
    }
    return result;
}

inline float quad_float(float a, float b, Overflow policy) {
    const double result = quad_fp(a, b);   // float -> double is exact

    const bool inputs_finite = std::isfinite(a) && std::isfinite(b);
    const bool overflowed    = inputs_finite && result > std::numeric_limits<float>::max();

    if (policy == Overflow::Clamp && overflowed) {
        return std::numeric_limits<float>::max();
    }
    return static_cast<float>(result);     // on IEEE platforms, too large -> +inf
}
```

- **Real infinities pass through.** Infinity only counts as overflow when both inputs were finite. If you pass in an infinity or NaN, you get it back in both modes.
- **Floats are computed in double.** A float has a 24-bit significand, so a product of two floats needs at most 48 bits and fits exactly in a double's 53. The largest possible result, about 3.5e77, is far below the double limit of 1.8e308. So for finite float inputs the fallback never runs; only infinite inputs reach it.

## Step 5: Process whole images fast

For images, `quad_image` processes whole buffers. On 10 million 8- to 16-bit pixels it takes 1–3 ms with `-O3 -march=native`, typically 2–5× faster than calling the single-value functions per pixel; float images gain less (see the table below).

Calling `quad_unsigned` once per pixel is slower for three reasons:

1. **The arithmetic is 64-bit** even for 8-bit data, so fewer pixels fit in each SIMD register.
2. **The policy and the bit width are re-checked** for every pixel.
3. **Branches block SIMD.** The compiler cannot easily process several pixels per instruction when each one goes through `if`s.

`quad_image` fixes all three.

### Integer images

```cpp
template <class Pixel>
void quad_image(const Pixel* a, const Pixel* b, Pixel* out, std::size_t count,
                unsigned bits, Overflow policy) {
    static_assert(std::is_integral_v<Pixel>, "use the float / double overloads for floating-point images");
    constexpr bool is_signed = std::is_signed_v<Pixel>;
    assert(bits >= (is_signed ? 2u : 1u) && bits <= 8 * sizeof(Pixel));

    // Arithmetic type. For pixels of up to 32 bits it has at least twice the
    // pixel's bits, so the exact result fits and modular arithmetic is exact.
    // For 64-bit pixels it is only as wide as the pixel: fine for Wrap, which
    // only needs the low bits; Clamp uses clamped_quad instead (see below).
    using Work = std::conditional_t<(sizeof(Pixel) <= 2), std::uint32_t, std::uint64_t>;

    const Work mask     = static_cast<Work>(low_bits_mask(bits));
    const Work max      = is_signed ? (mask >> 1) : mask;
    const Work sign_bit = max + 1;                   // used by signed wrap only

    if (policy == Overflow::Clamp && sizeof(Pixel) == 8) {
        // 64-bit pixels: the exact result can exceed 2^64, so use the
        // overflow-free checked version. Slower (it divides), but exact.
        for (std::size_t i = 0; i < count; ++i) {
            if constexpr (is_signed) {
                out[i] = static_cast<Pixel>(quad_signed(a[i], b[i], bits, Overflow::Clamp));
            } else {
                out[i] = static_cast<Pixel>(quad_unsigned(a[i], b[i], bits, Overflow::Clamp));
            }
        }
    } else if (policy == Overflow::Clamp) {
        for (std::size_t i = 0; i < count; ++i) {
            const Work x = static_cast<Work>(a[i]);  // negative values sign-extend
            const Work y = static_cast<Work>(b[i]);
            const Work result = x * x - x * y + y * y;               // exact, never negative
            out[i] = static_cast<Pixel>(result > max ? max : result);
        }
    } else if (!is_signed) {
        for (std::size_t i = 0; i < count; ++i) {
            const Work x = static_cast<Work>(a[i]);
            const Work y = static_cast<Work>(b[i]);
            out[i] = static_cast<Pixel>((x * x - x * y + y * y) & mask);
        }
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            const Work x = static_cast<Work>(a[i]);
            const Work y = static_cast<Work>(b[i]);
            const Work low_bits = (x * x - x * y + y * y) & mask;
            // Branch-free sign extension, same result as sign_extend():
            // for 8 bits, 0x81 -> (0x01 - 0x80) = ...FF81 = -127.
            const Work extended = (low_bits ^ sign_bit) - sign_bit;
            out[i] = static_cast<Pixel>(extended);
        }
    }
}
```

How it works:

- **Pixel type decides the details.** `Pixel` is the storage type (`uint8_t`, `uint16_t`, `int16_t`, ...), its signedness decides signed or unsigned, and `bits` is the real data width, e.g. 12 for 12-bit pixels in `uint16_t`.
- **The narrowest type that works.** Pixels of up to 16 bits give results below 2^32 (Fact 2), so `Work` is `uint32_t`; 32-bit pixels use `uint64_t`. For 64-bit pixels `uint64_t` is still enough for Wrap, which only needs the low bits.
- **64-bit Clamp is the one special case.** Its exact result can exceed 2^64, so that loop calls the checked single-value functions from Step 3. It is slower per pixel, but the quick exit in `clamped_quad` keeps it at about 2 ns.
- **Cast before multiplying.** `uint16_t * uint16_t` is computed as `int * int`, and 65535 × 65535 overflows a 32-bit `int`: undefined behaviour. Casting each pixel to `Work` first keeps all arithmetic unsigned.
- **One policy check per image.** Each policy gets its own loop, so the check is outside the loops.
- **Branch-free loop bodies.** `result > max ? max : result` compiles to a min instruction. For signed wrap, `sign_extend`'s `if` is replaced by the identity `(v ^ sign_bit) - sign_bit`: flipping the sign bit and subtracting it leaves positive values unchanged and makes values with the sign bit set negative.
- **In-place is allowed.** Each pixel is read before it is written, so `out` may be the same buffer as `a` or `b`.

### Float images

```cpp
inline void quad_image(const float* a, const float* b, float* out, std::size_t count,
                       Overflow policy) {
    const double inf       = std::numeric_limits<double>::infinity();
    const double float_max = std::numeric_limits<float>::max();
    const double limit     = (policy == Overflow::Clamp) ? float_max : inf;

    for (std::size_t i = 0; i < count; ++i) {
        const double x = a[i];
        const double y = b[i];
        const double result = x * x - x * y + y * y;
        const bool too_big = result > limit && result < inf;
        out[i] = static_cast<float>(too_big ? limit : result);
    }

    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(out[i])) {
            out[i] = quad_float(a[i], b[i], policy);
        }
    }
}
```

This splits `quad_float` into two passes so the main loop has no branches:

1. **Main pass:** the plain formula in double for every pixel, with the clamp as a select. For finite float inputs this is exactly what `quad_float` returns, because nothing can overflow in double (Step 4). For Wrap, `limit` is infinity, so nothing is replaced and the cast gives +inf for results above FLT\_MAX.
2. **Fix-up pass:** the only pixels that can be wrong are NaNs caused by infinite inputs, such as inf − inf. The second loop redoes just those with `quad_float`. Such pixels are rare, so the branch is almost never taken and the scan costs little.

The double overload simply calls `quad_double` in a loop. A two-pass version was slower for double in testing (32 ms versus 21 ms), because plain double arithmetic can genuinely overflow and the fix-up has more to do.

### Measured speed

Times in milliseconds for 10 million pixels, as Clamp / Wrap, best of 9 runs on one core of a 2.1 GHz Intel Xeon:

| Pixels | Per-pixel functions, `-O2` | `quad_image`, `-O2` | `quad_image`, `-O3 -march=native` |
| --- | --- | --- | --- |
| `uint8_t`, 8-bit | 10.9 / 11.0 | 9.0 / 8.6 | 3.4 / 1.1 |
| `uint16_t`, 12-bit | 11.0 / 11.0 | 6.6 / 2.3 | 2.8 / 2.3 |
| `int16_t`, 16-bit | 9.2 / 10.9 | 7.1 / 2.2 | 2.8 / 2.2 |
| `float` | 25.1 / 25.6 | 24.2 / 24.4 | 17.5 / 17.9 |
| `double` | 22.1 / 21.9 | 22.4 / 22.5 | 21.7 / 21.8 |

The float numbers vary somewhat between runs; an earlier run of the same code measured 15.8 / 13.4 ms with `-O3 -march=native`. 64-bit pixels with random values take about 21 ms for Clamp and 17–22 ms for Wrap, similar to the single-value functions.

For comparison, simply copying 10 million 16-bit pixels takes 1.5 ms, so the fastest integer cases are close to memory speed. Splitting the image across threads would help only a little there.

To get these numbers:

- **Build with `-O3 -march=native -DNDEBUG`.** `-O3` lets GCC vectorise the loops, `-march=native` enables your CPU's widest SIMD instructions, and `-DNDEBUG` removes the `assert`s. If the program will run on other machines, use a fixed target such as `-mavx2` instead of `-march=native`.
- **Check bit-exactness if it matters.** With `-march=native` the compiler may fuse multiply-adds, so double results can differ in the last bit between builds. Add `-ffp-contract=off` if you need bit-identical output everywhere.

## Step 6: The complete header

This is the full `quad_form.hpp`: every piece from Steps 1–5 in dependency order. Each function must appear before the functions that call it, so the helpers come first and `quad_image` last.

```cpp
// quad_form.hpp
//
// Computes  f(a, b) = a*a - a*b + b*b  for N-bit integers (N = 1..64),
// float and double, with a chosen overflow policy:
//   Overflow::Wrap  - keep the low N bits (like normal unsigned arithmetic)
//   Overflow::Clamp - saturate to the largest representable value
//
// Usage:
//   quad_unsigned(300, 100, 9, Overflow::Clamp);   // 9-bit unsigned  -> 511
//   quad_signed(-128, 127, 8, Overflow::Wrap);     // 8-bit signed    -> -127
//   quad_float(1.5f, 2.5f, Overflow::Clamp);
//   quad_double(1e200, 2e200, Overflow::Wrap);
//
//   // Whole images (much faster for bulk data):
//   quad_image(a, b, out, pixel_count, 12, Overflow::Clamp);   // 12-bit data in uint16_t
//   quad_image(fa, fb, fout, pixel_count, Overflow::Wrap);     // float image
//
// Requires C++17 (C++20 guarantees the two's-complement conversions).
// Uses only standard integer types up to 64 bits: no 128-bit integers.
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>

enum class Overflow { Wrap, Clamp };

// ===========================================================================
//  Helpers
// ===========================================================================

// Mask with the low `bits` bits set, e.g. 0x1FF for 9 bits.
// (Shifting right avoids 1 << 64, which is undefined.)
constexpr std::uint64_t low_bits_mask(unsigned bits) {
    return ~std::uint64_t{0} >> (64 - bits);
}

// Range of an N-bit integer.
constexpr std::uint64_t max_unsigned(unsigned bits) { return low_bits_mask(bits); }
constexpr std::int64_t  max_signed(unsigned bits)   { return static_cast<std::int64_t>(low_bits_mask(bits) >> 1); }
constexpr std::int64_t  min_signed(unsigned bits)   { return -max_signed(bits) - 1; }

// Treats the low `bits` bits of `value` as a two's-complement number and
// copies its sign bit into all higher bits.
constexpr std::uint64_t sign_extend(std::uint64_t value, unsigned bits) {
    const std::uint64_t sign_bit = std::uint64_t{1} << (bits - 1);

    const bool is_negative = (value & sign_bit) != 0;
    if (is_negative) {
        value |= ~low_bits_mask(bits);    // fill the upper bits with 1s
    }
    return value;
}

// a*a - a*b + b*b modulo 2^64.
//
// Negative inputs are passed in converted to uint64_t (-1 becomes 2^64 - 1).
// Unsigned arithmetic is defined as modulo 2^64, so the low bits of this
// result are always the low bits of the true result. When the true result
// is below 2^64 (inputs of up to 32 bits), this is the exact answer.
constexpr std::uint64_t quad_mod64(std::uint64_t x, std::uint64_t y) {
    return x * x - x * y + y * y;
}

// Adds x * y to `total` if the new total stays at most `limit`, and returns
// true. Returns false, leaving `total` unchanged, if it would exceed `limit`.
// Nothing here can overflow: the division tests x * y <= limit without
// computing x * y first.
constexpr bool add_product(std::uint64_t& total, std::uint64_t x, std::uint64_t y,
                           std::uint64_t limit) {
    if (x != 0 && y > limit / x) {
        return false;                       // x * y alone exceeds limit
    }
    const std::uint64_t product = x * y;
    if (product > limit - total) {
        return false;                       // total + product exceeds limit
    }
    total += product;
    return true;
}

// min(a*a - a*b + b*b, max), computed from |a| and |b| without overflow.
//
// With big = max(|a|, |b|) and small = min(|a|, |b|):
//   same sign:     big * (big - small) + small * small
//   opposite sign: big * big + big * small + small * small
// Every term is >= 0 and at most the true result, so the running total only
// grows, and as soon as it would pass `max` the answer is `max`.
constexpr std::uint64_t clamped_quad(std::uint64_t abs_a, std::uint64_t abs_b,
                                     bool same_sign, std::uint64_t max) {
    // Quick exit, no division: the result is never below (3/4) * big^2 (its
    // smallest value, reached at small = big / 2). Once big >= 2^33 that is
    // more than 2^64, so the result is certainly above max. (abs_a | abs_b)
    // has a bit at position 33 or higher exactly when big >= 2^33.
    if (((abs_a | abs_b) >> 33) != 0) {
        return max;
    }

    const std::uint64_t big   = abs_a > abs_b ? abs_a : abs_b;
    const std::uint64_t small = abs_a > abs_b ? abs_b : abs_a;

    std::uint64_t total = 0;
    if (same_sign) {
        if (!add_product(total, big, big - small, max)) return max;
    } else {
        if (!add_product(total, big, big,   max)) return max;
        if (!add_product(total, big, small, max)) return max;
    }
    if (!add_product(total, small, small, max)) return max;
    return total;
}

// |a| as an unsigned number. Works for INT64_MIN, whose magnitude 2^63 does
// not fit in int64_t (negating it as int64_t would be undefined).
constexpr std::uint64_t magnitude(std::int64_t a) {
    const std::uint64_t bits_of_a = static_cast<std::uint64_t>(a);
    return a < 0 ? 0 - bits_of_a : bits_of_a;
}

// ===========================================================================
//  Integers
// ===========================================================================

// a and b must be N-bit unsigned values: 0 <= a, b <= 2^bits - 1.
inline std::uint64_t quad_unsigned(std::uint64_t a, std::uint64_t b,
                                   unsigned bits, Overflow policy) {
    assert(bits >= 1 && bits <= 64);
    assert(a <= max_unsigned(bits) && b <= max_unsigned(bits));

    if (policy == Overflow::Wrap) {
        return quad_mod64(a, b) & low_bits_mask(bits);     // low `bits` bits
    }

    const std::uint64_t max = max_unsigned(bits);
    if (bits <= 32) {
        // The exact result is below 2^64, so quad_mod64 is exact: fast path.
        const std::uint64_t exact = quad_mod64(a, b);
        return exact > max ? max : exact;
    }
    return clamped_quad(a, b, /*same_sign=*/true, max);
}

// a and b must be N-bit signed values: -2^(bits-1) <= a, b <= 2^(bits-1) - 1.
inline std::int64_t quad_signed(std::int64_t a, std::int64_t b,
                                unsigned bits, Overflow policy) {
    assert(bits >= 2 && bits <= 64);
    assert(a >= min_signed(bits) && a <= max_signed(bits));
    assert(b >= min_signed(bits) && b <= max_signed(bits));

    // Converting a negative int64_t to uint64_t sign-extends it (modulo 2^64).
    const std::uint64_t x = static_cast<std::uint64_t>(a);
    const std::uint64_t y = static_cast<std::uint64_t>(b);

    if (policy == Overflow::Wrap) {
        const std::uint64_t low_bits = quad_mod64(x, y) & low_bits_mask(bits);
        return static_cast<std::int64_t>(sign_extend(low_bits, bits));
    }

    // The result is never negative, so clamping only needs an upper limit.
    const std::uint64_t max = static_cast<std::uint64_t>(max_signed(bits));
    if (bits <= 32) {
        // The exact result is below 3 * 2^62 < 2^64, so quad_mod64 is exact.
        const std::uint64_t exact = quad_mod64(x, y);
        return static_cast<std::int64_t>(exact > max ? max : exact);
    }
    const bool same_sign = (a < 0) == (b < 0);
    return static_cast<std::int64_t>(clamped_quad(magnitude(a), magnitude(b), same_sign, max));
}

// ===========================================================================
//  Floating point
// ===========================================================================
//
// Floats cannot wrap, so:
//   Overflow::Wrap  - standard IEEE behaviour: overflow gives +infinity
//   Overflow::Clamp - overflow gives the largest finite value
// Infinite or NaN inputs are passed through unchanged in both modes.

// Rearranged formula where every term is >= 0. No intermediate is larger than
// the final result, so nothing overflows unless the answer itself does.
inline double quad_all_terms_nonnegative(double a, double b) {
    const bool same_sign = (a >= 0) == (b >= 0);

    if (same_sign) {
        // ab >= 0, so both terms are >= 0.
        const double difference = a - b;
        return difference * difference + a * b;
    }

    // ab < 0, so subtracting it adds a positive amount.
    return a * a + b * b - a * b;
}

// The plain formula is accurate: |ab| <= (a^2 + b^2) / 2 <= result, so no term
// is more than twice the result and there is no harmful cancellation. Its one
// weakness is that an intermediate such as a*a can overflow (giving inf, or
// NaN from inf - inf) even when the answer is finite. That only happens for
// huge or infinite inputs, so we use the plain formula and fall back to the
// rearranged one in that rare case. This keeps the common path fast: choosing
// a formula by the signs of a and b for every pixel is a branch the CPU
// cannot predict on real image data.
inline double quad_fp(double a, double b) {
    const double plain = a * a - a * b + b * b;
    if (std::isfinite(plain)) {
        return plain;                              // almost always
    }
    return quad_all_terms_nonnegative(a, b);       // rare: overflow or infinite input
}

inline double quad_double(double a, double b, Overflow policy) {
    const double result = quad_fp(a, b);

    const bool inputs_finite = std::isfinite(a) && std::isfinite(b);
    const bool overflowed    = inputs_finite && std::isinf(result);

    if (policy == Overflow::Clamp && overflowed) {
        return std::numeric_limits<double>::max();
    }
    return result;
}

inline float quad_float(float a, float b, Overflow policy) {
    // Work in double: products of floats are exact there, and the double
    // range is large enough that this calculation cannot overflow.
    const double result = quad_fp(a, b);

    const bool inputs_finite = std::isfinite(a) && std::isfinite(b);
    const bool overflowed    = inputs_finite && result > std::numeric_limits<float>::max();

    if (policy == Overflow::Clamp && overflowed) {
        return std::numeric_limits<float>::max();
    }
    return static_cast<float>(result);   // on IEEE platforms, too large -> +inf
}

// ===========================================================================
//  Whole images
// ===========================================================================
//
// out[i] = f(a[i], b[i]) for every i in [0, count). `out` may be the same
// buffer as `a` or `b`.
//
// Integer images: Pixel is the storage type (uint8_t, uint16_t, int16_t,
// uint32_t, ...) and `bits` is the real width of the data, e.g. 12 for 12-bit
// pixels stored in uint16_t. Signedness comes from Pixel. Every pixel must
// already be within the `bits`-wide range; this is not checked per pixel.
//
// Why this is faster than calling quad_unsigned / quad_signed per pixel:
//   1. The arithmetic type is only as wide as needed. Pixels up to 16 bits
//      give results below 2^32, so 32-bit math is exact.
//   2. The policy is checked once per image, not once per pixel.
//   3. The loop bodies have no branches, so the compiler can use SIMD and
//      process many pixels per instruction. Build with -O3 and -march=native
//      (or e.g. -mavx2) to get the full benefit.
template <class Pixel>
void quad_image(const Pixel* a, const Pixel* b, Pixel* out, std::size_t count,
                unsigned bits, Overflow policy) {
    static_assert(std::is_integral_v<Pixel>, "use the float / double overloads for floating-point images");
    constexpr bool is_signed = std::is_signed_v<Pixel>;
    assert(bits >= (is_signed ? 2u : 1u) && bits <= 8 * sizeof(Pixel));

    // Arithmetic type. For pixels of up to 32 bits it has at least twice the
    // pixel's bits, so the exact result fits and modular arithmetic is exact.
    // For 64-bit pixels it is only as wide as the pixel: fine for Wrap, which
    // only needs the low bits; Clamp uses clamped_quad instead (see below).
    using Work = std::conditional_t<(sizeof(Pixel) <= 2), std::uint32_t, std::uint64_t>;

    const Work mask     = static_cast<Work>(low_bits_mask(bits));
    const Work max      = is_signed ? (mask >> 1) : mask;
    const Work sign_bit = max + 1;                   // used by signed wrap only

    if (policy == Overflow::Clamp && sizeof(Pixel) == 8) {
        // 64-bit pixels: the exact result can exceed 2^64, so use the
        // overflow-free checked version. Slower (it divides), but exact.
        for (std::size_t i = 0; i < count; ++i) {
            if constexpr (is_signed) {
                out[i] = static_cast<Pixel>(quad_signed(a[i], b[i], bits, Overflow::Clamp));
            } else {
                out[i] = static_cast<Pixel>(quad_unsigned(a[i], b[i], bits, Overflow::Clamp));
            }
        }
    } else if (policy == Overflow::Clamp) {
        for (std::size_t i = 0; i < count; ++i) {
            const Work x = static_cast<Work>(a[i]);  // negative values sign-extend
            const Work y = static_cast<Work>(b[i]);
            const Work result = x * x - x * y + y * y;               // exact, never negative
            out[i] = static_cast<Pixel>(result > max ? max : result);
        }
    } else if (!is_signed) {
        for (std::size_t i = 0; i < count; ++i) {
            const Work x = static_cast<Work>(a[i]);
            const Work y = static_cast<Work>(b[i]);
            out[i] = static_cast<Pixel>((x * x - x * y + y * y) & mask);
        }
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            const Work x = static_cast<Work>(a[i]);
            const Work y = static_cast<Work>(b[i]);
            const Work low_bits = (x * x - x * y + y * y) & mask;
            // Branch-free sign extension, same result as sign_extend():
            // for 8 bits, 0x81 -> (0x01 - 0x80) = ...FF81 = -127.
            const Work extended = (low_bits ^ sign_bit) - sign_bit;
            out[i] = static_cast<Pixel>(extended);
        }
    }
}

// Float images use two passes so the main loop has no branches and can be
// vectorised:
//   1. Plain formula in double for every pixel. For finite float inputs this
//      is exactly what quad_float computes: products of floats are exact in
//      double and nothing can overflow there.
//   2. The only pixels that can come out wrong are NaNs caused by infinite
//      inputs (e.g. inf - inf). Pass 2 redoes just those with quad_float.
//      This scan is cheap because such pixels are rare, so the branch is
//      almost never taken.
inline void quad_image(const float* a, const float* b, float* out, std::size_t count,
                       Overflow policy) {
    const double inf       = std::numeric_limits<double>::infinity();
    const double float_max = std::numeric_limits<float>::max();
    const double limit     = (policy == Overflow::Clamp) ? float_max : inf;

    for (std::size_t i = 0; i < count; ++i) {
        const double x = a[i];
        const double y = b[i];
        const double result = x * x - x * y + y * y;
        // Clamp: a finite result above FLT_MAX becomes FLT_MAX. Infinite
        // results only come from infinite inputs and pass through unchanged.
        // Wrap: limit is inf, so nothing is replaced; the cast gives +inf.
        const bool too_big = result > limit && result < inf;
        out[i] = static_cast<float>(too_big ? limit : result);
    }

    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(out[i])) {
            out[i] = quad_float(a[i], b[i], policy);
        }
    }
}

inline void quad_image(const double* a, const double* b, double* out, std::size_t count,
                       Overflow policy) {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = quad_double(a[i], b[i], policy);
    }
}
```

## Reproducing it: build and test

The full test suite compiles in one command and runs in about 2.4 seconds. It checks every input pair for widths up to 12 bits, the extreme values for 13 to 64 bits, `quad_image` for every pixel type and width, the checked clamp around its saturation point for every width from 33 to 64 bits, and the floating-point special cases.

### 1. Set up the folder

```
quad/
├── quad_form.hpp    // the header from Step 6
└── test_quad.cpp    // the test program below
```

The header works with any C++17 compiler, but the test program uses `__int128` for its reference answers (next section explains why), so build the tests with GCC 9+ or Clang 10+. On Windows, use MinGW-w64 or WSL.

### 2. Understand the test strategy

A good test compares against a reference that works *differently* from the code under test. If both used the same trick, a mistake in the trick would go unnoticed.

- **The header** relies on modular arithmetic in `uint64_t` (Fact 3) and, for clamping above 32 bits, on checked arithmetic that never forms a product larger than MAX.
- **The reference** uses neither. It works in 128-bit integers with the non-negative rearrangement from Step 4, (a − b)² + ab or a² + b² + |ab|, on magnitudes, so every step stays within \[0, 2^128) and is obviously exact. It then wraps or clamps by the plain definitions.

The header avoids 128-bit integers so it is portable; the reference uses them deliberately, because a reference should be as simple and obviously correct as possible, and it only has to run on the test machine.

Five kinds of tests use it:

- **Exhaustive** for widths up to 12 bits: every (a, b) pair, both policies. A 12-bit type has 4096 × 4096 = 16.7 million pairs.
- **Edge cases** for 13 to 64 bits: MIN, MIN+1, 0, 1, 2, MAX−1 and MAX in every combination. Overflow bugs almost always show up at these values.
- **`quad_image`** for every pixel type and width in the list: all edge pairs plus 100,000 random pairs each, compared with the reference.
- **Clamp boundary** for every width from 33 to 64 bits, signed and unsigned. Random values are almost always far above MAX, so this test concentrates on values where the result crosses MAX: around √MAX for same signs and √(MAX / 3) for opposite signs, plus 200,000 random pairs near those points. It checks both the single-value functions and `quad_image`.
- **Floating point:** explicit checks of overflow, infinity and NaN, and a check that the float `quad_image` matches `quad_float` exactly on special values plus 100,000 random pairs.

### 3. The test program

Save this as `test_quad.cpp`:

```cpp
// test_quad.cpp - checks quad_form.hpp against a simple exact reference.
#include "quad_form.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <random>
#include <type_traits>
#include <vector>

// The header uses no 128-bit integers. The reference below deliberately does:
// an independent, obviously-exact oracle is the point of a reference, and
// __int128 is available on GCC and Clang, where these tests run.
using i128    = __int128;
using uint128 = unsigned __int128;

// ---------- Reference: exact math that never overflows, then wrap / clamp ----------
//
// Deliberately different from the header: instead of relying on modular
// arithmetic, every step here stays within [0, 2^128), so nothing ever wraps.

uint128 exact_value(i128 a, i128 b) {
    const uint128 abs_a = static_cast<uint128>(a < 0 ? -a : a);
    const uint128 abs_b = static_cast<uint128>(b < 0 ? -b : b);
    const bool same_sign = (a < 0) == (b < 0);

    if (same_sign) {                                           // (a - b)^2 + ab
        const uint128 diff = abs_a > abs_b ? abs_a - abs_b : abs_b - abs_a;
        return diff * diff + abs_a * abs_b;
    }
    return abs_a * abs_a + abs_b * abs_b + abs_a * abs_b;      // a^2 + b^2 + |ab|
}

i128 reference(i128 a, i128 b, unsigned bits, bool is_signed, Overflow policy) {
    const uint128 exact = exact_value(a, b);
    const uint128 max   = is_signed ? (uint128{1} << (bits - 1)) - 1 : (uint128{1} << bits) - 1;

    if (policy == Overflow::Clamp) {
        return static_cast<i128>(exact > max ? max : exact);
    }
    i128 wrapped = static_cast<i128>(exact & ((uint128{1} << bits) - 1));   // low bits
    if (is_signed && wrapped > static_cast<i128>(max)) {
        wrapped -= i128{1} << bits;                                       // read as negative
    }
    return wrapped;
}

// ---------- Calls the function under test ----------

i128 compute(i128 a, i128 b, unsigned bits, bool is_signed, Overflow policy) {
    if (is_signed) {
        return quad_signed(static_cast<std::int64_t>(a), static_cast<std::int64_t>(b), bits, policy);
    }
    return quad_unsigned(static_cast<std::uint64_t>(a), static_cast<std::uint64_t>(b), bits, policy);
}

bool check(i128 a, i128 b, unsigned bits, bool is_signed) {
    for (Overflow policy : {Overflow::Wrap, Overflow::Clamp}) {
        if (compute(a, b, bits, is_signed, policy) != reference(a, b, bits, is_signed, policy)) {
            return false;
        }
    }
    return true;
}

i128 lowest(unsigned bits, bool is_signed)  { return is_signed ? min_signed(bits) : 0; }
i128 highest(unsigned bits, bool is_signed) {
    return is_signed ? i128{max_signed(bits)} : i128{max_unsigned(bits)};
}

void report(unsigned bits, bool is_signed, bool ok, const char* kind) {
    std::printf("%-8s %2u bits: %s%s\n", is_signed ? "signed" : "unsigned", bits,
                ok ? "OK" : "FAILED", kind);
}

// ---------- Exhaustive test: every pair (a, b) ----------

bool test_all_pairs(unsigned bits, bool is_signed) {
    const i128 lo = lowest(bits, is_signed);
    const i128 hi = highest(bits, is_signed);

    bool ok = true;
    for (i128 a = lo; a <= hi; ++a) {
        for (i128 b = lo; b <= hi; ++b) {
            if (!check(a, b, bits, is_signed)) ok = false;
        }
    }
    report(bits, is_signed, ok, "");
    return ok;
}

// ---------- Edge cases for widths too big to test exhaustively ----------

bool test_edge_cases(unsigned bits, bool is_signed) {
    const i128 lo = lowest(bits, is_signed);
    const i128 hi = highest(bits, is_signed);
    const i128 values[] = {lo, lo + 1, 0, 1, 2, hi - 1, hi};

    bool ok = true;
    for (i128 a : values) {
        for (i128 b : values) {
            if (!check(a, b, bits, is_signed)) ok = false;
        }
    }
    report(bits, is_signed, ok, " (edge cases)");
    return ok;
}

// ---------- Whole-image function: every edge pair plus 100,000 random pairs ----------

template <class Pixel>
bool test_image(unsigned bits) {
    constexpr bool is_signed = std::is_signed_v<Pixel>;
    const i128 lo = lowest(bits, is_signed);
    const i128 hi = highest(bits, is_signed);

    std::vector<i128> values;
    for (i128 v : {lo, lo + 1, i128{0}, i128{1}, i128{2}, hi - 1, hi}) {
        if (v >= lo && v <= hi) values.push_back(v);
    }

    std::vector<Pixel> a, b;
    for (i128 x : values) {
        for (i128 y : values) { a.push_back(static_cast<Pixel>(x)); b.push_back(static_cast<Pixel>(y)); }
    }
    std::mt19937_64 rng(bits);
    const i128 range = hi - lo + 1;
    for (int i = 0; i < 100000; ++i) {
        a.push_back(static_cast<Pixel>(lo + static_cast<i128>(rng()) % range));
        b.push_back(static_cast<Pixel>(lo + static_cast<i128>(rng()) % range));
    }

    bool ok = true;
    std::vector<Pixel> out(a.size());
    for (Overflow policy : {Overflow::Wrap, Overflow::Clamp}) {
        quad_image(a.data(), b.data(), out.data(), a.size(), bits, policy);
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (i128{out[i]} != reference(a[i], b[i], bits, is_signed, policy)) ok = false;
        }
    }
    report(bits, is_signed, ok, " (quad_image)");
    return ok;
}

// ---------- Clamp boundary for 33..64 bits ----------
//
// These widths use the checked (division-based) clamp. Most random pairs are
// far above MAX, so this test concentrates on values where the result crosses
// MAX: around sqrt(MAX) (same sign) and sqrt(MAX / 3) (opposite signs, a = -b).

i128 isqrt(i128 n) {
    i128 r = static_cast<i128>(std::sqrt(static_cast<long double>(n)));
    while (r * r > n) --r;
    while ((r + 1) * (r + 1) <= n) ++r;
    return r;
}

template <class Pixel>
bool test_clamp_boundary(unsigned bits) {
    constexpr bool is_signed = std::is_signed_v<Pixel>;
    const i128 lo = lowest(bits, is_signed);
    const i128 hi = highest(bits, is_signed);
    const i128 s  = isqrt(hi);
    const i128 t  = isqrt(hi / 3);

    std::vector<i128> values;
    for (i128 v : {lo, lo + 1, i128{0}, i128{1}, i128{2}, hi - 1, hi,
                   s - 1, s, s + 1, t - 1, t, t + 1}) {
        for (i128 w : {v, -v}) {
            if (w >= lo && w <= hi) values.push_back(w);
        }
    }

    std::vector<Pixel> a, b;
    for (i128 x : values) {
        for (i128 y : values) { a.push_back(static_cast<Pixel>(x)); b.push_back(static_cast<Pixel>(y)); }
    }
    std::mt19937_64 rng(bits * 7919);
    auto near = [&](i128 centre) {                  // random value in [centre/2, 2*centre]
        i128 v = centre / 2 + static_cast<i128>(rng() % static_cast<std::uint64_t>(centre * 3 / 2 + 1));
        if (is_signed && (rng() & 1)) v = -v;
        return v < lo ? lo : (v > hi ? hi : v);
    };
    for (int i = 0; i < 100000; ++i) {
        a.push_back(static_cast<Pixel>(near(s))); b.push_back(static_cast<Pixel>(near(s)));
        a.push_back(static_cast<Pixel>(near(t))); b.push_back(static_cast<Pixel>(near(t)));
    }

    bool ok = true;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!check(a[i], b[i], bits, is_signed)) ok = false;       // single-value functions
    }
    std::vector<Pixel> out(a.size());
    for (Overflow policy : {Overflow::Wrap, Overflow::Clamp}) {    // quad_image
        quad_image(a.data(), b.data(), out.data(), a.size(), bits, policy);
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (i128{out[i]} != reference(a[i], b[i], bits, is_signed, policy)) ok = false;
        }
    }
    report(bits, is_signed, ok, " (clamp boundary)");
    return ok;
}

// ---------- Floating-point special cases ----------

bool expect(bool condition, const char* description) {
    std::printf("%-58s %s\n", description, condition ? "OK" : "FAILED");
    return condition;
}

bool test_floating_point() {
    const double big      = 1.4e154;
    const double inf      = std::numeric_limits<double>::infinity();
    const double dbl_max  = std::numeric_limits<double>::max();
    const float  flt_max  = std::numeric_limits<float>::max();
    const float  finf     = std::numeric_limits<float>::infinity();

    bool ok = true;
    ok = expect(quad_float(3.0f, -2.0f, Overflow::Wrap) == 19.0f,              "float  3, -2 = 19") && ok;
    ok = expect(quad_double(big, big / 2, Overflow::Wrap) < dbl_max,          "double a*a overflows, answer finite -> finite") && ok;
    ok = expect(quad_double(big, big, Overflow::Wrap) == inf,                  "double true overflow, Wrap  -> +inf") && ok;
    ok = expect(quad_double(big, big, Overflow::Clamp) == dbl_max,             "double true overflow, Clamp -> DBL_MAX") && ok;
    ok = expect(quad_float(3e19f, 3e19f, Overflow::Wrap) == finf,              "float  true overflow, Wrap  -> +inf") && ok;
    ok = expect(quad_float(3e19f, 3e19f, Overflow::Clamp) == flt_max,          "float  true overflow, Clamp -> FLT_MAX") && ok;
    ok = expect(quad_float(finf, 1.0f, Overflow::Clamp) == finf,               "float  infinite input passes through") && ok;
    ok = expect(quad_double(inf, -inf, Overflow::Clamp) == inf,                "double +inf, -inf -> +inf") && ok;
    ok = expect(std::isnan(quad_float(std::nanf(""), 1.0f, Overflow::Clamp)),  "float  NaN input passes through") && ok;

    // quad_image must match quad_float exactly, including all the special cases.
    const float nan = std::nanf("");
    std::vector<float> fa = {3.0f, 3e19f, finf,  finf, -finf, finf, nan,  0.0f, -1.5f, 3e38f};
    std::vector<float> fb = {-2.0f, 3e19f, 1.0f, finf,  finf, 0.0f, 1.0f, 0.0f, 2.25f, -3e38f};
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1e20f, 1e20f);
    for (int i = 0; i < 100000; ++i) { fa.push_back(dist(rng)); fb.push_back(dist(rng)); }

    bool image_ok = true;
    std::vector<float> fo(fa.size());
    for (Overflow policy : {Overflow::Wrap, Overflow::Clamp}) {
        quad_image(fa.data(), fb.data(), fo.data(), fa.size(), policy);
        for (std::size_t i = 0; i < fa.size(); ++i) {
            const float expected = quad_float(fa[i], fb[i], policy);
            const bool same = (fo[i] == expected) || (std::isnan(fo[i]) && std::isnan(expected));
            if (!same) image_ok = false;
        }
    }
    ok = expect(image_ok, "float  quad_image matches quad_float (special + random)") && ok;
    return ok;
}

int main() {
    bool ok = true;

    // Exhaustive: all widths up to 12 bits (at most 16.7 million pairs each).
    for (unsigned bits : {1, 2, 3, 4, 8, 9, 10, 11, 12}) ok = test_all_pairs(bits, false) && ok;
    for (unsigned bits : {8, 9, 10, 11, 12})             ok = test_all_pairs(bits, true)  && ok;

    // Edge cases: MIN, MIN+1, 0, 1, 2, MAX-1, MAX in every combination.
    for (unsigned bits : {13, 14, 15, 16, 32, 64}) ok = test_edge_cases(bits, false) && ok;
    for (unsigned bits : {13, 14, 15, 16, 32, 64}) ok = test_edge_cases(bits, true)  && ok;

    // Whole-image function, for every pixel type and width.
    for (unsigned bits : {1, 2, 3, 4, 8}) ok = test_image<std::uint8_t>(bits)  && ok;
    for (unsigned bits = 9; bits <= 16; ++bits) ok = test_image<std::uint16_t>(bits) && ok;
    ok = test_image<std::uint32_t>(32) && ok;
    ok = test_image<std::uint64_t>(64) && ok;
    ok = test_image<std::int8_t>(8) && ok;
    for (unsigned bits = 9; bits <= 16; ++bits) ok = test_image<std::int16_t>(bits) && ok;
    ok = test_image<std::int32_t>(32) && ok;
    ok = test_image<std::int64_t>(64) && ok;

    // Checked clamp for 33..64 bits: values around the point where it saturates.
    for (unsigned bits = 33; bits <= 64; ++bits) ok = test_clamp_boundary<std::uint64_t>(bits) && ok;
    for (unsigned bits = 33; bits <= 64; ++bits) ok = test_clamp_boundary<std::int64_t>(bits)  && ok;

    // Floating point.
    ok = test_floating_point() && ok;

    std::printf(ok ? "\nALL TESTS PASSED\n" : "\nSOME TESTS FAILED\n");
    return ok ? 0 : 1;
}
```

### 4. Build and run

```bash
cd quad
g++ -std=c++20 -O2 -Wall -Wextra -o test_quad test_quad.cpp
./test_quad
```

This was tested with GCC under `-std=c++20` and `-std=c++17`, with `-O3 -march=native`, and with `-fsanitize=undefined,address` (no errors reported). Clang accepts the same code and flags; swap `g++` for `clang++`. Keep optimisation on: an unoptimised `-O0` build still passes but takes about 8 seconds instead of 2.4.

To confirm the header itself contains no 128-bit integers, compile any file that includes it with `-std=c++17 -pedantic -Werror`. That combination rejects `__int128`, and the header compiles cleanly under it.

### 5. Expected output

```
unsigned  1 bits: OK
unsigned  2 bits: OK
unsigned  3 bits: OK
unsigned  4 bits: OK
unsigned  8 bits: OK
unsigned  9 bits: OK
unsigned 10 bits: OK
unsigned 11 bits: OK
unsigned 12 bits: OK
signed    8 bits: OK
signed    9 bits: OK
signed   10 bits: OK
signed   11 bits: OK
signed   12 bits: OK
unsigned 13 bits: OK (edge cases)
unsigned 14 bits: OK (edge cases)
unsigned 15 bits: OK (edge cases)
unsigned 16 bits: OK (edge cases)
unsigned 32 bits: OK (edge cases)
unsigned 64 bits: OK (edge cases)
signed   13 bits: OK (edge cases)
signed   14 bits: OK (edge cases)
signed   15 bits: OK (edge cases)
signed   16 bits: OK (edge cases)
signed   32 bits: OK (edge cases)
signed   64 bits: OK (edge cases)
unsigned  1 bits: OK (quad_image)
unsigned  2 bits: OK (quad_image)
unsigned  3 bits: OK (quad_image)
unsigned  4 bits: OK (quad_image)
unsigned  8 bits: OK (quad_image)
unsigned  9 bits: OK (quad_image)
unsigned 10 bits: OK (quad_image)
unsigned 11 bits: OK (quad_image)
unsigned 12 bits: OK (quad_image)
unsigned 13 bits: OK (quad_image)
unsigned 14 bits: OK (quad_image)
unsigned 15 bits: OK (quad_image)
unsigned 16 bits: OK (quad_image)
unsigned 32 bits: OK (quad_image)
unsigned 64 bits: OK (quad_image)
signed    8 bits: OK (quad_image)
signed    9 bits: OK (quad_image)
signed   10 bits: OK (quad_image)
signed   11 bits: OK (quad_image)
signed   12 bits: OK (quad_image)
signed   13 bits: OK (quad_image)
signed   14 bits: OK (quad_image)
signed   15 bits: OK (quad_image)
signed   16 bits: OK (quad_image)
signed   32 bits: OK (quad_image)
signed   64 bits: OK (quad_image)
unsigned 33 bits: OK (clamp boundary)
unsigned 34 bits: OK (clamp boundary)
...                                      (one line per width, 35 to 63)
unsigned 64 bits: OK (clamp boundary)
signed   33 bits: OK (clamp boundary)
signed   34 bits: OK (clamp boundary)
...                                      (one line per width, 35 to 63)
signed   64 bits: OK (clamp boundary)
float  3, -2 = 19                                          OK
double a*a overflows, answer finite -> finite              OK
double true overflow, Wrap  -> +inf                        OK
double true overflow, Clamp -> DBL_MAX                     OK
float  true overflow, Wrap  -> +inf                        OK
float  true overflow, Clamp -> FLT_MAX                     OK
float  infinite input passes through                       OK
double +inf, -inf -> +inf                                  OK
float  NaN input passes through                            OK
float  quad_image matches quad_float (special + random)    OK

ALL TESTS PASSED
```

The full output has 126 OK lines; the `...` lines above stand for the 29 widths from 35 to 63 in each boundary block, which print in the same format.

The program exits with status 0 on success and 1 on any failure, so it can run in CI as is.

### 6. Extend the tests

- **More exhaustive widths:** each extra bit multiplies the pair count by 4. A 13-bit run takes about 1.7 seconds, so 16 bits takes roughly 100 seconds per signedness. Add `test_all_pairs(16, false)` and so on if you want them.
- **More random testing:** the boundary test's `near` helper is easy to reuse. Raise its 100,000 iterations, or add a third centre, to search more of the space around the saturation point.
- **Sanitizers:** keep a `-fsanitize=undefined,address` build in CI so any future change that introduces undefined behaviour is caught.

## Worked examples

Every value in this table was produced by running the header; working a couple by hand is the fastest way to check you understand it.

| Call | Exact a² − ab + b² | Wrap | Clamp |
| --- | --- | --- | --- |
| `quad_unsigned(1, 1, 1, …)` | 1 | 1 | 1 |
| `quad_unsigned(255, 0, 8, …)` | 65,025 | 1 | 255 |
| `quad_unsigned(300, 100, 9, …)` | 70,000 | 368 | 511 |
| `quad_signed(-128, 127, 8, …)` | 48,769 | −127 | 127 |
| `quad_signed(-5, 7, 12, …)` | 109 | 109 | 109 |
| `quad_signed(-2048, 2047, 12, …)` | 12,576,769 | −2047 | 2047 |
| `quad_signed(INT64_MIN, INT64_MAX, 64, …)` | ≈ 2.55 × 10³⁸ | −9,223,372,036,854,775,807 | 9,223,372,036,854,775,807 |

When the exact result fits (the −5, 7 row), Wrap and Clamp agree. They only differ on overflow. `quad_image` gives the same results for the same inputs.

### By hand: unsigned 9-bit, a = 300, b = 100

1. Exact: 90,000 − 30,000 + 10,000 = 70,000.
2. Wrap: 70,000 mod 2⁹ = 70,000 − 136 × 512 = 70,000 − 69,632 = **368**.
3. Clamp: 70,000 > 511, so **511**.

### By hand: signed 8-bit, a = −128, b = 127 (Wrap)

This traces `quad_signed(-128, 127, 8, Overflow::Wrap)` step by step. Values are 64-bit, in hex.

| Step | Value | Meaning |
| --- | --- | --- |
| `x = uint64_t(a)` | `0xFFFF'FFFF'FFFF'FF80` | −128 sign-extended mod 2^64 |
| `y = uint64_t(b)` | `0x7F` | 127 |
| `x * x` | `0x4000` | 16,384 = (−128)² |
| `x * y` | `0xFFFF'FFFF'FFFF'C080` | −16,256 mod 2^64 |
| `y * y` | `0x3F01` | 16,129 |
| `quad_mod64(x, y)` | `0xBE81` | 48,769, the exact answer (8 ≤ 32 bits) |
| `& 0xFF` | `0x81` | 129, the low 8 bits |
| `sign_extend(0x81, 8)` | `0xFFFF'FFFF'FFFF'FF81` | bit 7 is set, so fill with 1s |
| cast to `int64_t` | `-127` | final result |

Notice `x * y`: it "wrapped around" to a huge unsigned number, yet the final sum is still exactly 48,769. That is Fact 3 in action. Clamp for the same inputs does the same arithmetic, sees 48,769 > 127, and returns **127**.

`quad_image` with `int8_t` pixels does the same in 32 bits: `x` is `0xFFFF'FF80`, the sum is again `0xBE81`, and the branch-free sign extension gives (0x01 − 0x80) = `0xFFFF'FF81`, which is −127 as an `int8_t`.

### By hand: unsigned 40-bit Clamp (the checked path)

For 40 bits, MAX = 2^40 − 1 = 1,099,511,627,775, and `quad_unsigned` uses `clamped_quad` because 40 > 32. Unsigned values always have the same sign, so the terms are big · (big − small) and small².

**a = 1,000,000, b = 1** (true result 999,999,000,001, which fits):

1. Quick exit: 1,000,000 < 2^33, so the checked path runs.
2. big = 1,000,000, small = 1. First term: `add_product(total, 1000000, 999999, MAX)`. MAX / 1,000,000 = 1,099,511 (integer division), and 999,999 ≤ 1,099,511, so the product 999,999,000,000 fits. total = 999,999,000,000.
3. Second term: `add_product(total, 1, 1, MAX)`. 1 ≤ MAX − total, so total = **999,999,000,001**, the exact answer.

**a = 1,100,000, b = 1** (true result 1,209,998,900,001, above MAX):

1. Quick exit: 1,100,000 < 2^33, so the checked path runs.
2. First term: `add_product(total, 1100000, 1099999, MAX)`. MAX / 1,100,000 = 999,556, and 1,099,999 > 999,556, so the product alone exceeds MAX. The function returns **1,099,511,627,775** without ever computing the oversized product.

For comparison, Wrap for the second pair gives 1,209,998,900,001 mod 2^40 = **110,487,272,225**, computed directly with `quad_mod64`. All five values above were produced by running the header.

## Portability notes and pitfalls

The header uses only standard integer types up to 64 bits, so it works on any C++17 compiler; only the test program needs GCC or Clang.

### C++ version

- **C++17 is the minimum** because the header uses C++17 type traits such as `std::is_signed_v`.
- **C++20 is recommended.** Two places convert an unsigned value holding a sign-extended negative number to a signed type: the last line of `quad_signed` and the signed-wrap loop of `quad_image`. Before C++20 that conversion was implementation-defined; C++20 guarantees two's-complement wrapping. GCC, Clang and MSVC have always wrapped, so in practice C++17 gives the same results.

### MSVC and other compilers

The header needs no 128-bit integers and no extra libraries, so it should compile unchanged with MSVC or any other standard C++17 compiler. I checked this by compiling it with `-std=c++17 -pedantic -Werror`, which rejects `__int128` and other extensions, but I have not run it on MSVC itself.

The test program still uses `__int128` for its reference answers, so run the tests with GCC or Clang (on Windows, MinGW-w64 or WSL).

### Pitfalls this design avoids

| Pitfall | What goes wrong | How the code avoids it |
| --- | --- | --- |
| Integer promotion | `uint16_t * uint16_t` is computed as `int * int` and can overflow: undefined behaviour | Every value is cast to `uint64_t` or `Work` before any arithmetic |
| Signed overflow | `int64_t` arithmetic that overflows is undefined behaviour | All arithmetic is done in unsigned types |
| Overflow while checking for overflow | Computing x · y to compare it with MAX can itself overflow 64 bits | `add_product` tests y > MAX / x before multiplying |
| \|INT64\_MIN\| | `std::abs(INT64_MIN)` is undefined, because +2^63 does not fit in `int64_t` | `magnitude` negates in unsigned arithmetic |
| Shift by full width | `1ULL << 64` is undefined | Masks are built as `~0ULL >> (64 - bits)` |
| Clamping to MIN | Wasted code, and a sign that the math wasn't checked | The result is never negative (Fact 1), so only MAX is checked |
| Float intermediate overflow | a² overflows while the answer is finite, or inf − inf = NaN | Fallback to the rearranged formula whenever the plain result is not finite |
| Unpredictable branches | Branching on the signs or sizes of random values makes the CPU mispredict constantly: 78 ms vs 23 ms per 10 million floats, 61 ms vs 21 ms for 64-bit clamping | Plain formula first, branch-free image loops, and a branch-free quick exit in `clamped_quad` |
| Odd-width inputs out of range | A 9-bit value of 600 in a `uint16_t` gives wrong results | The single-value functions `assert`; `quad_image` requires valid pixels |

### Compiler flags

- **For speed, use `-O3 -march=native -DNDEBUG`,** or a fixed target such as `-mavx2` for binaries that run on other machines (Step 5).
- **Avoid `-ffast-math` / `/fp:fast`.** They let the compiler assume infinities and NaNs never occur and delete `std::isfinite` / `std::isnan` checks, which breaks the float fallback and the fix-up pass.
- **Add `-ffp-contract=off` for bit-identical doubles.** Otherwise `-march=native` may fuse multiply-adds and change the last bit between builds.
- **`-ftrapv` is harmless here.** It makes signed overflow abort, and the code never does signed arithmetic that overflows, but it will catch mistakes if you modify the code.

### Possible next steps

- **Threads for float and double images.** Splitting the image into stripes across threads (for example with OpenMP's `#pragma omp parallel for` over stripes, each calling `quad_image`) should help most for float and double, which are compute-bound. Integer images are already close to memory speed.
- **64-bit clamping without division.** If you clamp many 64-bit pixels whose values sit near the saturation point, the divisions in `add_product` become the main cost. A full 64 × 64-bit product can be built from four 32 × 32-bit multiplications into a high and a low 64-bit word, which avoids division while still using only standard types. For random or image-like data the quick exit already makes this unnecessary.
