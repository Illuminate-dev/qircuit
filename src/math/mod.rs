mod add;
mod div;
mod eq;
mod mul;
mod sub;

#[derive(Debug)]
pub enum Number {
    Complex(Complex),
    Real(Real),
}

impl Number {
    pub fn new_int(num: i64) -> Self {
        Number::Real(Real::new_integer(num))
    }

    pub fn new_complex(re: Real, im: Real) -> Self {
        Number::Complex(Complex::new(re, im))
    }
}

#[derive(Debug)]
pub struct Complex {
    pub re: Real,
    pub im: Real,
}

impl Complex {
    fn new(re: Real, im: Real) -> Self {
        Complex { re, im }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Real {
    Ratio(Ratio),
    Integer(i64),
}

impl Real {
    pub fn new_integer(num: i64) -> Self {
        Self::Integer(num)
    }

    pub fn new_ratio(num: i64, den: i64) -> Self {
        Self::Ratio(Ratio::new(num, den))
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Real::Ratio(a) => a.num == 0,
            Real::Integer(a) => *a == 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Ratio {
    pub num: i64,
    pub den: i64,
}

impl Ratio {
    pub fn new(num: i64, den: i64) -> Self {
        Self { num, den }
    }
}
