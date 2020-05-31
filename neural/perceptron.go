package neural

import (
	"github.com/TheBeege/go_neural_network_experiment/util"
	"math"
	"math/rand"
	"time"
)

type Perceptron struct {
	input        [][]float64
	actualOutput []float64
	weights      []float64
	bias         float64
	// epoch is one cycle of the training set
	epochs int
}

// Random Initialization
func (a *Perceptron) initialize() {
	rand.Seed(time.Now().UnixNano())
	a.bias = 0.0
	a.weights = make([]float64, len(a.input[0]))
	for i := 0; i < len(a.input[0]); i++ {
		a.weights[i] = rand.Float64()
	}
}

// Sigmoid Activation Function
func (a *Perceptron) sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Forward Propagation
// The dot product of the weight vector (w) and the input vector (x)
// is added with the bias (b) and the sum is passed through an activation function
func (a *Perceptron) forwardPass(x []float64) (sum float64) {
	return a.sigmoid(util.DotProduct(a.weights, x) + a.bias)
}

// We're using Mean Squared Error (MSE) for our loss function.
// MSE is often used for regression problems. There is also
// cross-entropy, but it's often used for classification problems

// Calculate Gradients of Weights
func (a *Perceptron) gradW(x []float64, y float64) []float64 {
	pred := a.forwardPass(x)
	return util.ScalarMatMul(-(pred-y)*pred*(1-pred), x)
}

// Calculate Gradients of Bias
func (a *Perceptron) gradB(x []float64, y float64) float64 {
	pred := a.forwardPass(x)
	return -(pred - y) * pred * (1 - pred)
}

// Train the Perceptron for n epochs
func (a *Perceptron) train() {
	for i := 0; i < a.epochs; i++ {
		dw := make([]float64, len(a.input[0]))
		db := 0.0
		for length, val := range a.input {
			dw = util.VecAdd(dw, a.gradW(val, a.actualOutput[length]))
			db += a.gradB(val, a.actualOutput[length])
		}
		dw = util.ScalarMatMul(2/float64(len(a.actualOutput)), dw)
		a.weights = util.VecAdd(a.weights, dw)
		a.bias += db * 2 / float64(len(a.actualOutput))
	}
}
