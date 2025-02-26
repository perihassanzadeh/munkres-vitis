/**********
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "munkres-cpp/src/munkres.h"

int main(int argc, char **argv) {
	int nrows = 5;
	int ncols = 5;
	
	//Read in File
	std::ifstream myfile("src/munkres_input_5x5.txt");

	//Create Matrix and populate values from file
	Matrix<double> matrix(nrows, ncols);
	for(int i=0; i<nrows; i++){
		for(int j=0; j<ncols; j++){
			double s;
			myfile >> s;
			matrix(i, j)= s;
		}
	}

	// Display begin matrix state.
	for ( int row = 0 ; row < nrows ; row++ ) {
		for ( int col = 0 ; col < ncols ; col++ ) {
			std::cout.width(2);
			std::cout << matrix(row,col) << ",";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	// Apply Munkres algorithm to matrix.
	Munkres<double> m;
	m.solve(matrix);


	// Display solved matrix.
	for ( int row = 0 ; row < nrows ; row++ ) {
		for ( int col = 0 ; col < ncols ; col++ ) {
			std::cout.width(2);
			std::cout << matrix(row,col) << ",";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	
	for ( int row = 0 ; row < nrows ; row++ ) {
		int rowcount = 0;
		for ( int col = 0 ; col < ncols ; col++  ) {
			if ( matrix(row,col) == 0 )
				rowcount++;
		}
		if ( rowcount != 1 )
			std::cerr << "Row " << row << " has " << rowcount << " columns that have been matched." << std::endl;
	}

	for ( int col = 0 ; col < ncols ; col++ ) {
		int colcount = 0;
		for ( int row = 0 ; row < nrows ; row++ ) {
			if ( matrix(row,col) == 0 )
				colcount++;
		}
		if ( colcount != 1 )
			std::cerr << "Column " << col << " has " << colcount << " rows that have been matched." << std::endl;
	}

	return 0;  
}
