/*
 *   Copyright (c) 2007 John Weaver
 *   Copyright (c) 2015 Miroslav Krajicek
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 */

#if !defined(_MUNKRES_H_)
#define _MUNKRES_H_

#include "matrix.h"

#include <list>
#include <utility>
#include <iostream>
#include <cmath>
#include <limits>

template<typename Data> class Munkres
{
    static constexpr int NORMAL = 0;
    static constexpr int STAR   = 1;
    static constexpr int PRIME  = 2;
public:

    /*
     *
     * Linear assignment problem solution
     * [modifies matrix in-place.]
     * matrix(row,col): row major format assumed.
     *
     * Assignments are remaining 0 values
     * (extra 0 values are replaced with -1)
     *
     */
    void solve(Matrix<Data> &m) {
        const int rows = m.rows(),
                columns = m.columns(),
                size = std::max(rows, columns);

#ifdef DEBUG
        std::cout << "Munkres input: " << m << std::endl;
#endif

        // Copy input matrix
        this->matrix = m;

        if ( rows != columns ) {
            // If the input matrix isn't square, make it square
            // and fill the empty values with the largest value present
            // in the matrix.
            matrix.resize(size, size, matrix.max());
        }


        // STAR == 1 == starred, PRIME == 2 == primed
        mask_matrix.resize(size, size);

        row_mask = new bool[size];
        col_mask = new bool[size];
        for ( int i = 0 ; i < size ; i++ ) {
            row_mask[i] = false;
        }

        for ( int i = 0 ; i < size ; i++ ) {
            col_mask[i] = false;
        }

        // Prepare the matrix values...

        // If there were any infinities, replace them with a value greater
        // than the maximum value in the matrix.
        replace_infinites(matrix);

        minimize_along_direction(matrix, rows >= columns);
        minimize_along_direction(matrix, rows <  columns);

        // Follow the steps
        int step = 1;
        while ( step ) {
            switch ( step ) {
            case 1:
                step = step1();
                // step is always 2
                break;
            case 2:
                step = step2();
                // step is always either 0 or 3
                break;
            case 3:
                step = step3();
                // step in [3, 4, 5]
                break;
            case 4:
                step = step4();
                // step is always 2
                break;
            case 5:
                step = step5();
                // step is always 3
                break;
            }
        }

        // Store results
        for ( int row = 0 ; row < size ; row++ ) {
            for ( int col = 0 ; col < size ; col++ ) {
                if ( mask_matrix(row, col) == STAR ) {
                    matrix(row, col) = 0;
                } else {
                    matrix(row, col) = -1;
                }
            }
        }

#ifdef DEBUG
        std::cout << "Munkres output: " << matrix << std::endl;
#endif
        // Remove the excess rows or columns that we added to fit the
        // input to a square matrix.
        matrix.resize(rows, columns);

        m = matrix;

        delete [] row_mask;
        delete [] col_mask;
    }

    static void replace_infinites(Matrix<Data> &matrix) {
      const int rows = matrix.rows(),
                columns = matrix.columns();
      assert( rows > 0 && columns > 0 );
      double max = matrix(0, 0);
      constexpr auto infinity = std::numeric_limits<double>::infinity();

      // Find the greatest value in the matrix that isn't infinity.
      for ( int row = 0 ; row < rows ; row++ ) {
        for ( int col = 0 ; col < columns ; col++ ) {
          if ( matrix(row, col) != infinity ) {
            if ( max == infinity ) {
              max = matrix(row, col);
            } else {
              max = std::max<double>(max, matrix(row, col));
            }
          }
        }
      }

      // a value higher than the maximum value present in the matrix.
      if ( max == infinity ) {
        // This case only occurs when all values are infinite.
        max = 0;
      } else {
        max++;
      }

      for ( int row = 0 ; row < rows ; row++ ) {
        for ( int col = 0 ; col < columns ; col++ ) {
          if ( matrix(row, col) == infinity ) {
            matrix(row, col) = max;
          }
        }
      }

    }

    static void minimize_along_direction(Matrix<Data> &matrix, const bool over_columns) {
      const int outer_size = over_columns ? matrix.columns() : matrix.rows(),
                   inner_size = over_columns ? matrix.rows() : matrix.columns();

      // Look for a minimum value to subtract from all values along
      // the "outer" direction.
      for ( int i = 0 ; i < outer_size ; i++ ) {
        double min = over_columns ? matrix(0, i) : matrix(i, 0);

        // As long as the current minimum is greater than zero,
        // keep looking for the minimum.
        // Start at one because we already have the 0th value in min.
        for ( int j = 1 ; j < inner_size && min > 0 ; j++ ) {
          min = std::min<double>(
            min,
            over_columns ? matrix(j, i) : matrix(i, j));
        }

        if ( min > 0 ) {
          for ( int j = 0 ; j < inner_size ; j++ ) {
            if ( over_columns ) {
              matrix(j, i) -= min;
            } else {
              matrix(i, j) -= min;
            }
          }
        }
      }
    }

private:

  inline bool find_uncovered_in_matrix(const double item, int &row, int &col) const {
    const int rows = matrix.rows(),
              columns = matrix.columns();

    for ( row = 0 ; row < rows ; row++ ) {
      if ( !row_mask[row] ) {
        for ( col = 0 ; col < columns ; col++ ) {
          if ( !col_mask[col] ) {
            if ( matrix(row,col) == item ) {
              return true;
            }
          }
        }
      }
    }

    return false;
  }

  bool pair_in_list(const std::pair<int,int> &needle, const std::list<std::pair<int,int> > &haystack) {
    for ( std::list<std::pair<int,int> >::const_iterator i = haystack.begin() ; i != haystack.end() ; i++ ) {
      if ( needle == *i ) {
        return true;
      }
    }

    return false;
  }

  int step1() {
    const int rows = matrix.rows(),
              columns = matrix.columns();

    for ( int row = 0 ; row < rows ; row++ ) {
      for ( int col = 0 ; col < columns ; col++ ) {
        if ( 0 == matrix(row, col) ) {
          for ( int nrow = 0 ; nrow < row ; nrow++ )
            if ( STAR == mask_matrix(nrow,col) )
              goto next_column;

          mask_matrix(row,col) = STAR;
          goto next_row;
        }
        next_column:;
      }
      next_row:;
    }

    return 2;
  }

  int step2() {
    const int rows = matrix.rows(),
              columns = matrix.columns();
    int covercount = 0;

    for ( int row = 0 ; row < rows ; row++ )
      for ( int col = 0 ; col < columns ; col++ )
        if ( STAR == mask_matrix(row, col) ) {
          col_mask[col] = true;
          covercount++;
        }

    if ( covercount >= matrix.minsize() ) {
  #ifdef DEBUG
      std::cout << "Final cover count: " << covercount << std::endl;
  #endif
      return 0;
    }

  #ifdef DEBUG
    std::cout << "Munkres matrix has " << covercount << " of " << matrix.minsize() << " Columns covered:" << std::endl;
    std::cout << matrix << std::endl;
  #endif


    return 3;
  }

  int step3() {
    /*
    Main Zero Search

     1. Find an uncovered Z in the distance matrix and prime it. If no such zero exists, go to Step 5
     2. If No Z* exists in the row of the Z', go to Step 4.
     3. If a Z* exists, cover this row and uncover the column of the Z*. Return to Step 3.1 to find a new Z
    */
    if ( find_uncovered_in_matrix(0, saverow, savecol) ) {
      mask_matrix(saverow,savecol) = PRIME; // prime it.
    } else {
      return 5;
    }

    for ( int ncol = 0 ; ncol < matrix.columns() ; ncol++ ) {
      if ( mask_matrix(saverow,ncol) == STAR ) {
        row_mask[saverow] = true; //cover this row and
        col_mask[ncol] = false; // uncover the column containing the starred zero
        return 3; // repeat
      }
    }

    return 4; // no starred zero in the row containing this primed zero
  }

  int step4() {
    const int rows = matrix.rows(),
              columns = matrix.columns();

    // seq contains pairs of row/column values where we have found
    // either a star or a prime that is part of the ``alternating sequence``.
    std::list<std::pair<int,int> > seq;
    // use saverow, savecol from step 3.
    std::pair<int,int> z0(saverow, savecol);
    seq.insert(seq.end(), z0);

    // We have to find these two pairs:
    std::pair<int,int> z1(-1, -1);
    std::pair<int,int> z2n(-1, -1);

    int row, col = savecol;
    /*
    Increment Set of Starred Zeros

     1. Construct the ``alternating sequence'' of primed and starred zeros:

           Z0 : Unpaired Z' from Step 4.2
           Z1 : The Z* in the column of Z0
           Z[2N] : The Z' in the row of Z[2N-1], if such a zero exists
           Z[2N+1] : The Z* in the column of Z[2N]

        The sequence eventually terminates with an unpaired Z' = Z[2N] for some N.
    */
    bool madepair;
    do {
      madepair = false;
      for ( row = 0 ; row < rows ; row++ ) {
        if ( mask_matrix(row,col) == STAR ) {
          z1.first = row;
          z1.second = col;
          if ( pair_in_list(z1, seq) ) {
            continue;
          }

          madepair = true;
          seq.insert(seq.end(), z1);
          break;
        }
      }

      if ( !madepair )
        break;

      madepair = false;

      for ( col = 0 ; col < columns ; col++ ) {
        if ( mask_matrix(row, col) == PRIME ) {
          z2n.first = row;
          z2n.second = col;
          if ( pair_in_list(z2n, seq) ) {
            continue;
          }
          madepair = true;
          seq.insert(seq.end(), z2n);
          break;
        }
      }
    } while ( madepair );

    for ( std::list<std::pair<int,int> >::iterator i = seq.begin() ;
        i != seq.end() ;
        i++ ) {
      // 2. Unstar each starred zero of the sequence.
      if ( mask_matrix(i->first,i->second) == STAR )
        mask_matrix(i->first,i->second) = NORMAL;

      // 3. Star each primed zero of the sequence,
      // thus increasing the number of starred zeros by one.
      if ( mask_matrix(i->first,i->second) == PRIME )
        mask_matrix(i->first,i->second) = STAR;
    }

    // 4. Erase all primes, uncover all columns and rows,
    for ( int row = 0 ; row < mask_matrix.rows() ; row++ ) {
      for ( int col = 0 ; col < mask_matrix.columns() ; col++ ) {
        if ( mask_matrix(row,col) == PRIME ) {
          mask_matrix(row,col) = NORMAL;
        }
      }
    }

    for ( int i = 0 ; i < rows ; i++ ) {
      row_mask[i] = false;
    }

    for ( int i = 0 ; i < columns ; i++ ) {
      col_mask[i] = false;
    }

    // and return to Step 2.
    return 2;
  }

  int step5() {
    const int rows = matrix.rows(),
              columns = matrix.columns();
    /*
    New Zero Manufactures

     1. Let h be the smallest uncovered entry in the (modified) distance matrix.
     2. Add h to all covered rows.
     3. Subtract h from all uncovered columns
     4. Return to Step 3, without altering stars, primes, or covers.
    */
    double h = std::numeric_limits<double>::max();
    for ( int row = 0 ; row < rows ; row++ ) {
      if ( !row_mask[row] ) {
        for ( int col = 0 ; col < columns ; col++ ) {
          if ( !col_mask[col] ) {
            if ( h > matrix(row, col) && matrix(row, col) != 0 ) {
              h = matrix(row, col);
            }
          }
        }
      }
    }

    for ( int row = 0 ; row < rows ; row++ ) {
      if ( row_mask[row] ) {
        for ( int col = 0 ; col < columns ; col++ ) {
          matrix(row, col) += h;
        }
      }
    }

    for ( int col = 0 ; col < columns ; col++ ) {
      if ( !col_mask[col] ) {
        for ( int row = 0 ; row < rows ; row++ ) {
          matrix(row, col) -= h;
        }
      }
    }

    return 3;
  }

  Matrix<int> mask_matrix;
  Matrix<Data> matrix;
  bool *row_mask;
  bool *col_mask;
  int saverow = 0, savecol = 0;
};


#endif /* !defined(_MUNKRES_H_) */
