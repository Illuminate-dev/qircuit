use std::ops::Sub;

use super::{Ratio, Real};

// NUMBER

// COMPLEX

// REAL
impl Sub for Real {
    type Output = Real;
    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Real::Integer(a), Real::Integer(b)) => Real::Integer(a - b),
            (Real::Integer(a), Real::Ratio(b)) => Real::Ratio(a - b),
            (Real::Ratio(a), Real::Integer(b)) => Real::Ratio(a - b),
            (Real::Ratio(a), Real::Ratio(b)) => Real::Ratio(a - b),
        }
    }
}

// RATIO

impl Sub<i64> for Ratio {
    type Output = Ratio;
    fn sub(self, rhs: i64) -> Self::Output {
        Ratio::new(self.num - rhs * self.den, self.den)
    }
}

impl Sub<Ratio> for i64 {
    type Output = Ratio;
    fn sub(self, rhs: Ratio) -> Self::Output {
        Ratio::new(self * rhs.den - rhs.num, rhs.den)
    }
}

impl Sub<Ratio> for Ratio {
    type Output = Ratio;
    fn sub(self, rhs: Ratio) -> Self::Output {
        Ratio::new(self.num * rhs.den - rhs.num * self.den, self.den * rhs.den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_real() {
        assert_eq!(
            Real::new_integer(1) - Real::new_integer(2),
            Real::new_integer(-1)
        );
        assert_ne!(
            Real::new_integer(1) - Real::new_integer(2),
            Real::new_integer(2)
        );

        assert_eq!(
            Real::new_integer(1) - Real::new_ratio(1, 2),
            Real::new_ratio(1, 2)
        );
        assert_ne!(
            Real::new_integer(1) - Real::new_ratio(1, 2),
            Real::new_ratio(3, 3)
        );

        assert_eq!(
            Real::new_ratio(1, 2) - Real::new_ratio(0, 2),
            Real::new_ratio(1, 2)
        );

        assert_eq!(
            Real::new_ratio(1, 2) - Real::new_ratio(0, -2),
            Real::new_ratio(1, 2)
        );

        assert_eq!(
            Real::new_ratio(1, 2) - Real::new_ratio(-0, -2),
            Real::new_ratio(1, 2)
        );

        assert_eq!(
            Real::new_ratio(7, 8) - Real::new_ratio(1, 3),
            Real::new_ratio(13, 24)
        );
    }

    #[test]
    fn add_ratio_ratio() {
        assert_eq!(Ratio::new(0, 1) - Ratio::new(0, 1), Ratio::new(0, 1));
        assert_eq!(Ratio::new(0, 1) - Ratio::new(1, 1), Ratio::new(-1, 1));

        assert_eq!(Ratio::new(1, 2) - Ratio::new(0, 2), Ratio::new(1, 2));

        assert_eq!(Ratio::new(1, 2) - Ratio::new(0, -2), Ratio::new(1, 2));

        assert_eq!(Ratio::new(1, 2) - Ratio::new(-0, -2), Ratio::new(1, 2));

        assert_eq!(Ratio::new(1, 3) - Ratio::new(7, 8), Ratio::new(-13, 24));
    }
}
