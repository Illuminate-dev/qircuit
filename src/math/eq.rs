use super::{Complex, Number, Ratio, Real};

// NUMBER
impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Number::Real(a), Number::Real(b)) => a == b,
            (Number::Real(a), Number::Complex(b)) => a == b,
            (Number::Complex(a), Number::Real(b)) => a == b,
            (Number::Complex(a), Number::Complex(b)) => a == b,
        }
    }
}

// COMPLEX

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}

impl PartialEq<Real> for Complex {
    fn eq(&self, other: &Real) -> bool {
        self.re == *other && self.im.is_zero()
    }
}

impl PartialEq<Complex> for Real {
    fn eq(&self, other: &Complex) -> bool {
        other.re == *self && other.im.is_zero()
    }
}

// REAL

impl PartialEq for Real {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Real::Integer(a), Real::Integer(b)) => a == b,
            (Real::Integer(a), Real::Ratio(b)) => a == b,
            (Real::Ratio(a), Real::Integer(b)) => a == b,
            (Real::Ratio(a), Real::Ratio(b)) => a == b,
        }
    }
}

// RATIO

impl PartialEq for Ratio {
    fn eq(&self, other: &Self) -> bool {
        self.num * other.den == other.num * self.den
    }
}

impl PartialEq<i64> for Ratio {
    fn eq(&self, other: &i64) -> bool {
        self.num == *other * self.den
    }
}

impl PartialEq<Ratio> for i64 {
    fn eq(&self, other: &Ratio) -> bool {
        *self * other.den == other.num
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eq_number() {
        assert_eq!(Number::new_int(1), Number::new_int(1));
        assert_ne!(Number::new_int(1), Number::new_int(0));
        assert_eq!(
            Number::new_complex(Real::new_integer(7), Real::new_integer(5)),
            Number::new_complex(Real::new_integer(7), Real::new_integer(5))
        );
        assert_ne!(
            Number::new_complex(Real::new_integer(7), Real::new_integer(5)),
            Number::new_complex(Real::new_integer(5), Real::new_integer(7))
        );

        assert_eq!(
            Number::new_int(1),
            Number::new_complex(Real::new_integer(1), Real::new_integer(0))
        );
        assert_ne!(
            Number::new_int(1),
            Number::new_complex(Real::new_integer(1), Real::new_integer(1))
        );
        assert_eq!(
            Number::new_complex(Real::new_integer(1), Real::new_integer(0)),
            Number::new_int(1)
        );
        assert_ne!(
            Number::new_complex(Real::new_integer(1), Real::new_integer(1)),
            Number::new_int(1)
        );
    }

    #[test]
    fn test_eq_complex() {
        assert_eq!(
            Complex::new(Real::new_integer(1), Real::new_integer(1)),
            Complex::new(Real::new_integer(1), Real::new_integer(1))
        );

        assert_ne!(
            Complex::new(Real::new_integer(0), Real::new_integer(1)),
            Complex::new(Real::new_integer(1), Real::new_integer(1))
        );

        assert_ne!(
            Complex::new(Real::new_integer(1), Real::new_integer(1)),
            Complex::new(Real::new_integer(1), Real::new_integer(0))
        );
    }

    #[test]
    fn test_eq_real() {
        assert_eq!(Real::new_integer(1), Real::new_integer(1));
        assert_ne!(Real::new_integer(1), Real::new_integer(0));

        assert_eq!(Real::new_ratio(1, 2), Real::new_ratio(1, 2));
        assert_ne!(Real::new_ratio(1, 2), Real::new_ratio(1, 3));
        assert_ne!(Real::new_ratio(1, 2), Real::new_ratio(3, 2));
        assert_ne!(Real::new_ratio(1, 2), Real::new_ratio(3, 3));

        assert_eq!(Real::new_ratio(3, 3), Real::new_ratio(1, 1));
        assert_eq!(Real::new_ratio(3, 3), Real::new_integer(1));
        assert_ne!(Real::new_ratio(3, 3), Real::new_integer(3));
    }

    #[test]
    fn test_eq_ratio() {
        assert_eq!(Ratio::new(1, 2), Ratio::new(1, 2));
        assert_ne!(Ratio::new(1, 2), Ratio::new(1, 3));
        assert_ne!(Ratio::new(1, 2), Ratio::new(3, 2));
        assert_ne!(Ratio::new(1, 2), Ratio::new(3, 3));
    }
}
