package util

// Dot Product of Two Vectors of same size
func DotProduct(v1, v2 []float64) float64 {
	dot := 0.0
	for i := 0; i < len(v1); i++ {
		dot += v1[i] * v2[i]
	}
	return dot
}

// Addition of Two Vectors of same size
func VecAdd(v1, v2 []float64) []float64 {
	add := make([]float64, len(v1))
	for i := 0; i < len(v1); i++ {
		add[i] = v1[i] + v2[i]
	}
	return add
}

// Multiplication of a Vector & Matrix
func ScalarMatMul(s float64, mat []float64) []float64 {
	result := make([]float64, len(mat))
	for i := 0; i < len(mat); i++ {
		result[i] += s * mat[i]
	}
	return result
}
