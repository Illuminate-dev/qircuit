use std::ops::Div;

use super::{Complex, Number, Ratio, Real};

// NUMBER
impl Div for Number {
    type Output = Number;
    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Number::Real(a), Number::Real(b)) => Number::Real(a / b),
            (Number::Real(a), Number::Complex(b)) => Number::Complex(a / b),
            (Number::Complex(a), Number::Real(b)) => Number::Complex(a / b),
            (Number::Complex(a), Number::Complex(b)) => Number::Complex(a / b),
        }
    }
}

// COMPLEX
impl Div<Complex> for Complex {
    type Output = Complex;
    fn div(self, rhs: Self) -> Self::Output {
        Complex::new(
            (self.re * rhs.re + self.im * rhs.im) / (rhs.re * rhs.re + rhs.im * rhs.im),
            (self.re * rhs.im - self.im * rhs.re) / (rhs.re * rhs.re + rhs.im * rhs.im),
        )
    }
}

impl Div<Real> for Complex {
    type Output = Complex;
    fn div(self, rhs: Real) -> Self::Output {
        Complex::new(self.re / rhs, self.im / rhs)
    }
}

impl Div<Complex> for Real {
    type Output = Complex;
    fn div(self, rhs: Complex) -> Self::Output {
        Complex::new(self / rhs.re, self / rhs.im)
    }
}

// REAL
impl Div for Real {
    type Output = Real;
    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Real::Integer(a), Real::Integer(b)) => Real::new_ratio(a, b),
            (Real::Integer(a), Real::Ratio(b)) => Real::Ratio(a / b),
            (Real::Ratio(a), Real::Integer(b)) => Real::Ratio(a / b),
            (Real::Ratio(a), Real::Ratio(b)) => Real::Ratio(a / b),
        }
    }
}

// RATIO
impl Div<Ratio> for Ratio {
    type Output = Ratio;
    fn div(self, rhs: Self) -> Self::Output {
        Ratio::new(self.num * rhs.den, self.den * rhs.num)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<i64> for Ratio {
    type Output = Ratio;
    fn div(self, rhs: i64) -> Self::Output {
        Ratio::new(self.num, self.den * rhs)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<Ratio> for i64 {
    type Output = Ratio;
    fn div(self, rhs: Ratio) -> Self::Output {
        Ratio::new(self * rhs.den, rhs.num)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_num() {
        assert_eq!(
            Number::Real(Real::new_integer(1)) / Number::Real(Real::new_integer(2)),
            Number::Real(Real::new_ratio(1, 2))
        );

        assert_ne!(
            Number::Real(Real::new_integer(1)) / Number::Real(Real::new_integer(2)),
            Number::Real(Real::new_integer(1))
        );

        assert_eq!(
            Number::new_int(5) / Number::new_complex(Real::new_integer(1), Real::new_integer(1)),
            Number::new_complex(Real::new_integer(5), Real::new_integer(5))
        );
        assert_ne!(
            Number::new_int(2) / Number::new_complex(Real::new_integer(1), Real::new_integer(1)),
            Number::new_complex(Real::new_integer(5), Real::new_integer(5))
        );

        assert_eq!(
            Number::new_complex(Real::new_integer(1), Real::new_integer(1)) / Number::new_int(5),
            Number::new_complex(Real::new_ratio(1, 5), Real::new_ratio(1, 5))
        );
        assert_ne!(
            Number::new_complex(Real::new_integer(1), Real::new_integer(1)) / Number::new_int(2),
            Number::new_complex(Real::new_integer(5), Real::new_integer(5))
        );

        assert_eq!(
            Number::new_complex(Real::new_integer(1), Real::new_integer(1))
                / Number::new_complex(Real::new_integer(1), Real::new_integer(1)),
            Number::new_complex(Real::new_integer(1), Real::new_integer(0))
        );

        assert_eq!(
            Number::new_complex(Real::new_integer(5), Real::new_integer(1))
                / Number::new_complex(Real::new_integer(1), Real::new_integer(5)),
            Number::new_complex(Real::new_ratio(5, 13), Real::new_ratio(12, 13))
        );
    }

    #[test]
    fn test_ratio_ratio() {
        assert_eq!(Ratio::new(1, 2) / Ratio::new(1, 2), Ratio::new(1, 1));
        assert_eq!(Ratio::new(1, 2) / 2, Ratio::new(1, 4));
    }
}
