use crate::{Error, Vm};
pub mod candle;
// pub mod math; // Раскомментируйте, когда перенесете математику

pub trait NativeModule {
    fn name(&self) -> &str;
    fn register(&self, vm: &mut Vm) -> Result<(), Error>;
}

pub fn available_extensions() -> Vec<Box<dyn NativeModule>> {
    vec![
        Box::new(candle::CandleExtension),
        // Box::new(math::MathExtension),
    ]
}
