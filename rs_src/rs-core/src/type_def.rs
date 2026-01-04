use std::ops::Add;

#[derive(Debug, Default)]
pub struct FheInfo {
    rotation: u32,
    mul_both: u32,
    mul_single: u32,
}

impl FheInfo {
    fn new(rotation: u32, mul_both: u32, mul_single: u32) -> Self {
        FheInfo {
            rotation,
            mul_both,
            mul_single,
        }
    }
}

impl Add for FheInfo {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl Add<&Self> for FheInfo {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let rotation = self.rotation + rhs.rotation;
        let mul_both = self.mul_both + rhs.mul_both;
        let mul_single = self.mul_single + rhs.mul_single;

        Self {
            rotation,
            mul_both,
            mul_single,
        }
    }
}
