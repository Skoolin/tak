use std::{error::Error, path::Path};

use tak::game::Game;
use tch::{
    data::Iter2,
    nn,
    nn::{ConvConfig, ModuleT, OptimizerConfig},
    Device,
    Kind,
    Tensor,
};

use crate::{
    example::Example,
    repr::{game_repr, input_dims, moves_dims},
};

const EPOCHS: usize = 1; // probably don't want to look at one position more then once, except with applied symmetries
const BATCH_SIZE: i64 = 10_000; // just do whole dataset https://discord.com/channels/176389490762448897/932765511358513204/934552328479080479
const LEARNING_RATE: f64 = 1e-3;
const WEIGHT_DECAY: f64 = 1e-4;
const MIN_LOSS_DIFF: f32 = 0.2;

#[derive(Debug)]
pub struct Network<const N: usize> {
    vs: nn::VarStore,
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    conv3: nn::Conv2D,
    conv4: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
    fc4: nn::Linear,
}

impl<const N: usize> Network<N> {
    pub fn predict(&self, game: &Game<N>, train: bool) -> (Tensor, Tensor) {
        let input = game_repr(game);
        let output = self.forward_t(&input.unsqueeze(0), train);
        let mut vec = output.split(moves_dims(N) as i64, 1);
        let eval = vec.pop().unwrap();
        let policy = vec.pop().unwrap();
        (policy, eval)
    }

    pub fn train(&mut self, examples: &[Example<N>]) {
        println!("starting training with {} examples", examples.len());
        let mut opt = nn::Adam {
            wd: WEIGHT_DECAY,
            ..Default::default()
        }
        .build(&self.vs, LEARNING_RATE)
        .unwrap();

        let games: Vec<_> = examples
            .iter()
            .map(|Example { game, .. }| game_repr(game))
            .collect();
        let targets: Vec<_> = examples
            .iter()
            .map(|Example { pi, v, .. }| Tensor::cat(&[pi, v], 0))
            .collect();

        let mut last_loss = 0.0;
        for epoch in 0..EPOCHS {
            // Batch examples
            let mut batch_iter =
                Iter2::new(&Tensor::stack(&games, 0), &Tensor::stack(&targets, 0), BATCH_SIZE);
            let batch_iter = batch_iter
                .to_device(Device::cuda_if_available())
                .return_smaller_last_batch()
                .shuffle(); // (Looks like shuffle has hardcoded CPU)

            println!("epoch: {}", epoch);
            let mut epoch_loss = 0.;
            for (input, target) in batch_iter {
                let batch_size = input.size()[0];
                let output = self.forward_t(&input, true);
                // get prediction
                let mut vec = output.split(moves_dims(N) as i64, 1);
                let eval = vec.pop().unwrap();
                let policy = vec.pop().unwrap();

                // Get target
                let mut vec = target.split(moves_dims(N) as i64, 1);
                let z = vec.pop().unwrap();
                let p = vec.pop().unwrap();

                let loss_p = -(p * policy).sum(Kind::Float) / batch_size;
                let loss_z = (z - eval).square().sum(Kind::Float) / batch_size;
                let total_loss = loss_z + loss_p;

                opt.backward_step(&total_loss);
                epoch_loss += {
                    let x: f32 = total_loss.into();
                    x
                };
            }
            println!("loss: {:?}", epoch_loss);

            // quit early if loss diff is too small
            let loss_diff = (last_loss - epoch_loss).abs();
            if loss_diff < MIN_LOSS_DIFF {
                break;
            }
            last_loss = epoch_loss;
        }
    }

    pub fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        self.vs.save(path)?;
        Ok(())
    }

    pub fn load<T: AsRef<Path>>(path: T) -> Result<Network<N>, Box<dyn Error>> {
        let mut nn = Self::default();
        nn.vs.load(path)?;
        Ok(nn)
    }
}

impl<const N: usize> Default for Network<N> {
    fn default() -> Self {
        // TODO make sure dimensions work for any board size
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let root = &vs.root();
        let [d1, _d2, _d3] = input_dims(N);
        let conv1 = nn::conv2d(root, d1 as i64, 32, 3, ConvConfig {
            padding: 1,
            ..Default::default()
        });
        let conv2 = nn::conv2d(root, 32, 64, 3, ConvConfig {
            padding: 1,
            ..Default::default()
        });
        let conv3 = nn::conv2d(root, 64, 128, 3, ConvConfig {
            padding: 1,
            ..Default::default()
        });
        let conv4 = nn::conv2d(root, 128, 128, 3, ConvConfig {
            padding: 1,
            ..Default::default()
        });
        let fc1 = nn::linear(root, (N * N * 128) as i64, 2048, Default::default());
        let fc2 = nn::linear(root, 2048, 1024, Default::default());
        let fc3 = nn::linear(root, 1024, moves_dims(N) as i64, Default::default());
        let fc4 = nn::linear(root, 1024, 1, Default::default());
        Network {
            vs,
            conv1,
            conv2,
            conv3,
            conv4,
            fc1,
            fc2,
            fc3,
            fc4,
        }
    }
}

impl<const N: usize> nn::ModuleT for Network<N> {
    fn forward_t(&self, input: &Tensor, _train: bool) -> Tensor {
        let s = input
            .apply(&self.conv1)
            .apply(&self.conv2)
            .apply(&self.conv3)
            .apply(&self.conv4)
            .reshape(&[-1, (N * N * 128) as i64])
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
            .relu();
        let policy = s.apply(&self.fc3).log_softmax(1, Kind::Float);
        let eval = s.apply(&self.fc4).tanh();
        // would be nice if I could just return two values
        Tensor::cat(&[policy, eval], 1)
    }
}
