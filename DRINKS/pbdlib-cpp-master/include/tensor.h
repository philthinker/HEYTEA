/*
 * tensor.h
 *
 * N-dimensional tensor class, built around a flattened 1-D vector
 *
 * This class only contains enough functionalities to implement the demos, since
 * armadillo doesn't provide tensors with more than 3 dimensions
 *
 * Authors: Philip Abbet
 */

#pragma once


#include <stdio.h>
#include <assert.h>
#include <armadillo>


template <typename T>
class Tensor {

public:

	//-------------------------------------------------------------------------
	// Creates an empty tensor
	//-------------------------------------------------------------------------
	Tensor()
	: size(0)
	{
	}


	//-------------------------------------------------------------------------
	// Creates the tensor with the provided dimensions
	//-------------------------------------------------------------------------
	template<typename... Ints>
	Tensor(Ints... dims)
	: Tensor(arma::ivec({((int) dims)...}))
	{
	}


	//-------------------------------------------------------------------------
	// Creates the tensor with the provided dimensions
	//-------------------------------------------------------------------------
	Tensor(const arma::ivec& dims)
	: size(1)
	{
		this->dims = dims;

		for (int i = 0; i < dims.n_elem; ++i)
			size *= this->dims(i);

		_indices = arma::ivec(dims.n_elem);

		_indices(0) = 1;
		for (int i = 1; i < dims.n_elem; ++i)
			_indices(i) = dims(i - 1) * _indices(i - 1);

		data = arma::Col<T>(size, arma::fill::zeros);
	}


	//-------------------------------------------------------------------------
	// Returns the element index (in the flattened vector) corresponding to the
	// provided subscript indices
	//-------------------------------------------------------------------------
	int indice(const arma::ivec& subscripts) const {
		assert(subscripts.n_elem == dims.n_elem);

		int indice = 0;

		for (int i = 0; i < dims.n_elem; ++i)
			indice += subscripts(i) * _indices(i);

		return indice;
	}


	//-------------------------------------------------------------------------
	// Returns the element index (in the flattened vector) corresponding to the
	// provided subscript indices
	//-------------------------------------------------------------------------
	template<typename... Ints>
	int indice(Ints... subscripts) const {
		assert(sizeof...(Ints) == dims.n_elem);

		return indice(arma::ivec({subscripts...}));
	}


	//-------------------------------------------------------------------------
	// Returns the list of element indices (in the flattened vector)
	// corresponding to the provided subscript indices
	//-------------------------------------------------------------------------
	template<typename... Spans>
	arma::uvec indices(Spans... spans) const {
		assert(sizeof...(Spans) == dims.n_elem);

		std::vector<arma::span> all_spans(dims.n_elem);
		_process_spans(all_spans, 0, spans...);

		arma::ivec subscripts(dims.n_elem);
		int nb_indices = 1;

		for (int i = 0; i < dims.n_elem; ++i) {
			subscripts(i) = all_spans[i].a;
			nb_indices *= all_spans[i].b - all_spans[i].a + 1;
		}

		arma::uvec result(nb_indices);
		for (int i = 0; i < nb_indices; ++i) {
			result(i) = indice(subscripts);

			int j = 0;

			while (j < dims.n_elem) {
				subscripts(j) += 1;
				if (subscripts(j) > all_spans[j].b) {
					subscripts(j) = all_spans[j].a;
					++j;
				} else {
					break;
				}
			}
		}

		return result;
	}


	//-------------------------------------------------------------------------
	// Returns A COPY of the elements located at the provided indices
	//-------------------------------------------------------------------------
	inline const arma::Col<T> operator()(const arma::uvec& indices) const {
		return data(indices);
	}


	//-------------------------------------------------------------------------
	// Returns A COPY of the elements located at the provided indices/spans
	//-------------------------------------------------------------------------
	template<typename... Spans>
	inline Tensor<T> operator()(Spans... spans) const {
		assert(sizeof...(Spans) == dims.n_elem);

		arma::ivec dims(this->dims.n_elem);

		_compute_dims(dims, 0, spans...);

		Tensor<T> result(dims);
		result.data = data(indices(spans...));
		return result;
	}


	//-------------------------------------------------------------------------
	// Returns the value of the n-th element under the assumption of a flat
	// layout, with column-major ordering of data (ie. column by column)
	//-------------------------------------------------------------------------
	inline T operator[](int index) const {
		return data(index);
	}


	//-------------------------------------------------------------------------
	// Assignement operator
	//-------------------------------------------------------------------------
	inline void operator=(const arma::Col<T>& v) {
		assert(size == v.n_elem);
		data = v;
	}


	//-------------------------------------------------------------------------
	// Assignement operator
	//-------------------------------------------------------------------------
	inline void operator=(const Tensor<T>& t) {
		assert(size == t.size);
		assert(arma::all(dims == t.dims));
		data = t.data;
	}


	//-------------------------------------------------------------------------
	// Set the value at the location given by provided index
	//-------------------------------------------------------------------------
	inline void set(int index, T value) {
		data(index) = value;
	}


	//-------------------------------------------------------------------------
	// Set the value at the locations given by provided indices
	//-------------------------------------------------------------------------
	inline void set(const arma::uvec& indices, T value) {
		for (int i = 0; i < indices.n_elem; ++i)
			data(indices(i)) = value;
	}


	//-------------------------------------------------------------------------
	// Set the values at the locations given by provided indices
	//-------------------------------------------------------------------------
	inline void set(const arma::uvec& indices, const arma::Col<T>& values) {
		data(indices) = values;
	}


	//-------------------------------------------------------------------------
	// Set the value at the locations given by provided indices
	//-------------------------------------------------------------------------
	template<typename... Spans>
	inline void set(Spans... spans, T value) {
		static_assert(sizeof...(Spans) == dims.n_elem, "Wrong number of arguments");

		data(indices(spans...)) = arma::Col<T>({ value });
	}


	//-------------------------------------------------------------------------
	// Set the values at the locations given by provided indices
	//-------------------------------------------------------------------------
	template<typename... Spans>
	inline void set(Spans... spans, const arma::Col<T>& values) {
		static_assert(sizeof...(Spans) == dims.n_elem, "Wrong number of arguments");

		data(indices(spans...)) = values;
	}


	//-------------------------------------------------------------------------
	// Multiplication by a scalar
	//-------------------------------------------------------------------------
	Tensor<T> operator*(T scalar) const {
		Tensor<T> result(dims);
		result.data = data * scalar;
		return result;
	}


	//-------------------------------------------------------------------------
	// Division by a scalar
	//-------------------------------------------------------------------------
	Tensor<T> operator/(T scalar) const {
		Tensor<T> result(dims);
		result.data = data / scalar;
		return result;
	}


	//-------------------------------------------------------------------------
	// Addition of a scalar
	//-------------------------------------------------------------------------
	Tensor<T> operator+(T scalar) const {
		Tensor<T> result(dims);
		result.data = data + scalar;
		return result;
	}


	//-------------------------------------------------------------------------
	// Addition of a tensor of same dimensions
	//-------------------------------------------------------------------------
	Tensor<T> operator+(const Tensor<T>& m) const {
		Tensor<T> result(dims);
		result.data = data + m.data;
		return result;
	}


	//-------------------------------------------------------------------------
	// Substraction of a tensor of same dimensions
	//-------------------------------------------------------------------------
	Tensor<T> operator-(const Tensor<T>& m) const {
		Tensor<T> result(dims);
		result.data = data - m.data;
		return result;
	}


protected:

	inline void _process_span(arma::span &s, unsigned int size) const {
		if (s.whole) {
			s.a = 0;
			s.b = size - 1;
		}
	}


	template<typename... Spans>
	void _process_spans(std::vector<arma::span> &result, unsigned int dim,
						int index, Spans... spans) const {

		result[dim] = arma::span(index);
		_process_spans(result, dim + 1, spans...);
	}


	template<typename... Spans>
	void _process_spans(std::vector<arma::span> &result, unsigned int dim,
						arma::span indices, Spans... spans) const {

		_process_span(indices, dims[dim]);
		result[dim] = indices;
		_process_spans(result, dim + 1, spans...);
	}


	void _process_spans(std::vector<arma::span> &result, unsigned int dim,
						int index) const {

		result[dim] = arma::span(index);
	}


	void _process_spans(std::vector<arma::span> &result, unsigned int dim,
						arma::span indices) const {

		_process_span(indices, dims[dim]);
		result[dim] = indices;
	}


	template<typename... Spans>
	void _compute_dims(arma::ivec &result, unsigned int dim, arma::span indices,
					   Spans... spans) const {
		_process_span(indices, dims(dim));

		result[dim] = indices.b - indices.a + 1;

		_compute_dims(result, dim + 1, spans...);
	}


	void _compute_dims(arma::ivec &result, unsigned int dim, arma::span indices) const {
		_process_span(indices, dims(dim));

		result[dim] = indices.b - indices.a + 1;
	}


public:
	arma::ivec		dims;	// The N dimensions of the tensor
	int				size;	// Total number of elements
	arma::Col<T>	data;	// The flattened 1-D vector

protected:
	arma::ivec		_indices;	// Used to compute 1-D indices from N-D ones
};


typedef Tensor<double>      TensorDouble;
typedef Tensor<float>       TensorFloat;
typedef Tensor<arma::sword> TensorInt;
typedef Tensor<arma::uword> TensorUInt;

