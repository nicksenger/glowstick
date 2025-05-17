use std::ops::{Add, Mul};

use glowstick::dynamic::Wild;
use glowstick::num::{U1000, U151, U936};
use glowstick::Dyn;

pub type L = Dyn<Wild>;
pub type Bhl = Dyn<Wild>;
pub type U151936 = <<U151 as Mul<U1000>>::Output as Add<U936>>::Output;
