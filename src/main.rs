use ndarray::prelude::*;
use ndarray::stack;
use ndarray::Array;

fn main() {
    let x = array![1., 2., 4.];
    let x0 = Array::<f64, _>::ones(x.len());
    let phi = stack![Axis(1), x0, x];
    let mut w = array![0., 0.];
    let y = array![1., 3., 3.];

    let preds = phi.dot(&w);

    dbg!(&phi);
    dbg!(&preds);
    let loss = preds - &y;
    let sq_loss = loss.mapv(|a| a.powi(2));
    dbg!(&loss);
    dbg!(&sq_loss);
    let train_loss = sq_loss.sum() / y.len() as f64;
    dbg!(train_loss);
    //let grad_train_loss = ((2. / x.len() as f64) * (loss * phi));
    let grad_train_loss = ((phi * loss.into_shape((3, 1)).unwrap()) * 2.) // convert loss to a
        // column vector then
        // multiply. Also
        // multiply by 2 for the
        // derivative
        .mean_axis(Axis(0)); // Sum along the columns
    dbg!(&grad_train_loss);
    // Next need to update the weights
    let eta = 0.1;
    w = w - eta * grad_train_loss.unwrap();
    dbg!(&w);
}
