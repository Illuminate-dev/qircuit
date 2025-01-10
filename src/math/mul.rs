use std::ops::Mul;

use super::{Complex, Number, Ratio, Real};

// NUMBER
impl Mul for Number {
    type Output = Number;
    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Number::Real(a), Number::Real(b)) => Number::Real(a * b),
            (Number::Real(a), Number::Complex(b)) => Number::Complex(a * b),
            (Number::Complex(a), Number::Real(b)) => Number::Complex(a * b),
            (Number::Complex(a), Number::Complex(b)) => Number::Complex(a * b),
        }
    }
}

// COMPLEX
impl Mul<Complex> for Complex {
    type Output = Complex;
    fn mul(self, rhs: Self) -> Self::Output {
        Complex::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Mul<Real> for Complex {
    type Output = Complex;
    fn mul(self, rhs: Real) -> Self::Output {
        Complex::new(self.re * rhs, self.im * rhs)
    }
}

impl Mul<Complex> for Real {
    type Output = Complex;
    fn mul(self, rhs: Complex) -> Self::Output {
        Complex::new(self * rhs.re, self * rhs.im)
    }
}

// REAL
impl Mul for Real {
    type Output = Real;
    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Real::Integer(a), Real::Integer(b)) => Real::Integer(a * b),
            (Real::Integer(a), Real::Ratio(b)) => Real::Ratio(a * b),
            (Real::Ratio(a), Real::Integer(b)) => Real::Ratio(a * b),
            (Real::Ratio(a), Real::Ratio(b)) => Real::Ratio(a * b),
        }
    }
}

// RATIO
impl Mul<Ratio> for Ratio {
    type Output = Ratio;
    fn mul(self, rhs: Self) -> Self::Output {
        Ratio::new(self.num * rhs.num, self.den * rhs.den)
    }
}

impl Mul<i64> for Ratio {
    type Output = Ratio;
    fn mul(self, rhs: i64) -> Self::Output {
        Ratio::new(self.num * rhs, self.den)
    }
}

impl Mul<Ratio> for i64 {
    type Output = Ratio;
    fn mul(self, rhs: Ratio) -> Self::Output {
        Ratio::new(self * rhs.num, rhs.den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_num() {
        assert_eq!(
            Number::Real(Real::new_integer(1)) * Number::Real(Real::new_integer(2)),
            Number::Real(Real::new_integer(2))
        );

        assert_ne!(
            Number::Real(Real::new_integer(1)) * Number::Real(Real::new_integer(2)),
            Number::Real(Real::new_integer(1))
        );

        assert_eq!(
            Number::new_int(5) * Number::new_complex(Real::new_integer(1), Real::new_integer(1)),
            Number::new_complex(Real::new_integer(5), Real::new_integer(5))
        );
        assert_ne!(
            Number::new_int(2) * Number::new_complex(Real::new_integer(1), Real::new_integer(1)),
            Number::new_complex(Real::new_integer(5), Real::new_integer(5))
        );

        assert_eq!(
            Number::new_complex(Real::new_integer(1), Real::new_integer(1)) * Number::new_int(5),
            Number::new_complex(Real::new_integer(5), Real::new_integer(5))
        );
        assert_ne!(
            Number::new_complex(Real::new_integer(1), Real::new_integer(1)) * Number::new_int(2),
            Number::new_complex(Real::new_integer(5), Real::new_integer(5))
        );

        assert_eq!(
            Number::new_complex(Real::new_integer(1), Real::new_integer(1))
                * Number::new_complex(Real::new_integer(1), Real::new_integer(1)),
            Number::new_complex(Real::new_integer(0), Real::new_integer(2))
        );

        assert_eq!(
            Number::new_complex(Real::new_integer(1), Real::new_integer(1))
                * Number::new_complex(Real::new_integer(1), Real::new_integer(1)),
            Number::new_complex(Real::new_integer(0), Real::new_integer(2))
        );
    }

    #[test]
    fn test_ratio_ratio() {
        assert_eq!(Ratio::new(1, 2) * Ratio::new(1, 2), Ratio::new(1, 4));
        assert_eq!(Ratio::new(1, 2) * 2, Ratio::new(1, 1));
    }
}
