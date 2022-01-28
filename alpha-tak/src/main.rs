use network::Network;
use self_play::play_until_better;
use tch::{Cuda, Tensor, CModule, TchError};
use tch;

#[macro_use]
extern crate lazy_static;

pub mod example;
pub mod mcts;
pub mod network;
pub mod repr;
pub mod self_play;
pub mod turn_map;

const START: usize = 0;

fn main() -> Result<(), TchError> {
    tch::maybe_init_cuda();
    println!("CUDA: {}", Cuda::is_available());

    let opts = (tch::Kind::Float, tch::Device::Cpu);

    let model = CModule::load("forward_10_128.pt")?;
    let input = tch::IValue::Tensor(Tensor::zeros(&[28, 6, 6], opts));

    let output = model.forward_is(&[input]);
    println!("{:?}", output);
    Ok(())
}
