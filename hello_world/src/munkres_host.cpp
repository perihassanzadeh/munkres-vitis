#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <sstream>

#include "CL/cl.h"
#include "CL/cl_ext_xilinx.h"
#include "xcl2.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <limits>

constexpr int NORMAL = 0;
constexpr int STAR   = 1;
constexpr int PRIME  = 2;
constexpr size_t MAX_SIZE = 256;

// Add global variables after the existing constants
double global_matrix[MAX_SIZE][MAX_SIZE];
int global_mask_matrix[MAX_SIZE][MAX_SIZE];
bool global_row_mask[MAX_SIZE];
bool global_col_mask[MAX_SIZE];
size_t global_size;
size_t global_saverow;
size_t global_savecol;
double global_item;  

// Function declarations
void replace_infinites(void);
void minimize_along_direction(bool over_columns);
bool find_uncovered_in_matrix(size_t &row, size_t &col);  // Modified to remove item parameter
int step1(void);
int step2(void);
int step3(void);
int step4(void);
int step5(void);

// Function implementations
void replace_infinites(void) {
    double max = global_matrix[0][0];
    constexpr auto infinity = std::numeric_limits<double>::infinity();

    // Find the greatest value in the matrix that isn't infinity
    for (size_t row = 0; row < global_size; row++) {
        for (size_t col = 0; col < global_size; col++) {
            if (global_matrix[row][col] != infinity) {
                if (max == infinity) {
                    max = global_matrix[row][col];
                } else {
                    max = std::max<double>(max, global_matrix[row][col]);
                }
            }
        }
    }

    // A value higher than the maximum value present in the matrix
    if (max == infinity) {
        max = 0;
    } else {
        max++;
    }

    for (size_t row = 0; row < global_size; row++) {
        for (size_t col = 0; col < global_size; col++) {
            if (global_matrix[row][col] == infinity) {
                global_matrix[row][col] = max;
            }
        }
    }
}

void minimize_along_direction(double matrix[][MAX_SIZE], size_t size, bool over_columns) {
    for (size_t i = 0; i < size; i++) {
        double min = over_columns ? matrix[0][i] : matrix[i][0];

        for (size_t j = 1; j < size && min > 0; j++) {
            min = std::min<double>(min, over_columns ? matrix[j][i] : matrix[i][j]);
        }

        if (min > 0) {
            for (size_t j = 0; j < size; j++) {
                if (over_columns) {
                    matrix[j][i] -= min;
                } else {
                    matrix[i][j] -= min;
                }
            }
        }
    }
}

bool find_uncovered_in_matrix_hls(const double matrix[][MAX_SIZE], const int mask_matrix[][MAX_SIZE],
                                const bool row_mask[], const bool col_mask[],
                                const double item, size_t &row, size_t &col, size_t size,
                                cl::Context& context, cl::CommandQueue& q, cl::Kernel& krnl_find_uncovered) {
    try {
        cl_int err;
        
        // Flatten 2D arrays for buffer creation with aligned allocator
        std::vector<double, aligned_allocator<double>> flat_matrix(MAX_SIZE * MAX_SIZE);
        std::vector<int, aligned_allocator<int>> flat_mask_matrix(MAX_SIZE * MAX_SIZE);
        std::vector<bool, aligned_allocator<bool>> flat_row_mask(MAX_SIZE);
        std::vector<bool, aligned_allocator<bool>> flat_col_mask(MAX_SIZE);
        
        for (size_t i = 0; i < MAX_SIZE; i++) {
            for (size_t j = 0; j < MAX_SIZE; j++) {
                flat_matrix[i * MAX_SIZE + j] = matrix[i][j];
                flat_mask_matrix[i * MAX_SIZE + j] = mask_matrix[i][j];
            }
            flat_row_mask[i] = row_mask[i];
            flat_col_mask[i] = col_mask[i];
        }
        
        // Allocate memory on device with initial data
        cl::Buffer matrix_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            sizeof(double)*MAX_SIZE*MAX_SIZE, flat_matrix.data());
        cl::Buffer mask_matrix_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            sizeof(int)*MAX_SIZE*MAX_SIZE, flat_mask_matrix.data());
        cl::Buffer row_mask_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            sizeof(bool)*MAX_SIZE, flat_row_mask.data());
        cl::Buffer col_mask_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            sizeof(bool)*MAX_SIZE, flat_col_mask.data());
        cl::Buffer row_out_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(size_t), nullptr);
        cl::Buffer col_out_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(size_t), nullptr);
        cl::Buffer found_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(bool), nullptr);

        // Set kernel arguments
        OCL_CHECK(krnl_find_uncovered.setArg(0, matrix_buffer),
                  "Failed to set matrix buffer argument");
        OCL_CHECK(krnl_find_uncovered.setArg(1, mask_matrix_buffer),
                  "Failed to set mask matrix buffer argument");
        OCL_CHECK(krnl_find_uncovered.setArg(2, row_mask_buffer),
                  "Failed to set row mask buffer argument");
        OCL_CHECK(krnl_find_uncovered.setArg(3, col_mask_buffer),
                  "Failed to set column mask buffer argument");
        OCL_CHECK(krnl_find_uncovered.setArg(4, item),
                  "Failed to set item argument");
        OCL_CHECK(krnl_find_uncovered.setArg(5, size),
                  "Failed to set size argument");
        OCL_CHECK(krnl_find_uncovered.setArg(6, row_out_buffer),
                  "Failed to set row output buffer argument");
        OCL_CHECK(krnl_find_uncovered.setArg(7, col_out_buffer),
                  "Failed to set column output buffer argument");
        OCL_CHECK(krnl_find_uncovered.setArg(8, found_buffer),
                  "Failed to set found buffer argument");

        // Migrate input data to device
        std::vector<cl::Memory> inBufVec;
        inBufVec.push_back(matrix_buffer);
        inBufVec.push_back(mask_matrix_buffer);
        inBufVec.push_back(row_mask_buffer);
        inBufVec.push_back(col_mask_buffer);
        OCL_CHECK(q.enqueueMigrateMemObjects(inBufVec, 0 /* 0 means from host*/),
                  "Failed to migrate input buffers to device");

        // Launch kernel
        cl::Event event;
        OCL_CHECK(q.enqueueTask(krnl_find_uncovered, nullptr, &event),
                  "Failed to launch kernel");

        // Migrate output data to host
        std::vector<cl::Memory> outBufVec;
        outBufVec.push_back(found_buffer);
        outBufVec.push_back(row_out_buffer);
        outBufVec.push_back(col_out_buffer);
        OCL_CHECK(q.enqueueMigrateMemObjects(outBufVec, CL_MIGRATE_MEM_OBJECT_HOST),
                  "Failed to migrate output buffers to host");

        // Wait for all operations to complete
        OCL_CHECK(q.finish(),
                  "Failed to finish command queue");

        // Get kernel execution time
        cl_ulong time_start, time_end;
        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
        double duration = (time_end - time_start) / 1000000.0; // Convert to milliseconds
        std::cout << "Kernel execution time: " << duration << " ms" << std::endl;

        // Get results from migrated buffers
        bool found;
        size_t row_out, col_out;
        void* found_ptr = q.enqueueMapBuffer(found_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(bool), nullptr, nullptr, nullptr);
        void* row_ptr = q.enqueueMapBuffer(row_out_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(size_t), nullptr, nullptr, nullptr);
        void* col_ptr = q.enqueueMapBuffer(col_out_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(size_t), nullptr, nullptr, nullptr);
        
        found = *static_cast<bool*>(found_ptr);
        row_out = *static_cast<size_t*>(row_ptr);
        col_out = *static_cast<size_t*>(col_ptr);

        // Unmap buffers
        q.enqueueUnmapMemObject(found_buffer, found_ptr);
        q.enqueueUnmapMemObject(row_out_buffer, row_ptr);
        q.enqueueUnmapMemObject(col_out_buffer, col_ptr);

        if (found) {
            row = row_out;
            col = col_out;
            return true;
        }
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in find_uncovered_in_matrix_hls: " << e.what() << std::endl;
        return false;
    }
}

int step1(void) {
    for (size_t row = 0; row < global_size; row++) {
        bool found_star = false;
        for (size_t col = 0; col < global_size && !found_star; col++) {
            if (global_matrix[row][col] == 0) {
                bool has_star = false;
                for (size_t nrow = 0; nrow < row && !has_star; nrow++) {
                    if (global_mask_matrix[nrow][col] == STAR) {
                        has_star = true;
                    }
                }
                if (!has_star) {
                    global_mask_matrix[row][col] = STAR;
                    found_star = true;
                }
            }
        }
    }
    return 2;
}

int step2(void) {
    size_t covercount = 0;
    for (size_t row = 0; row < global_size; row++) {
        for (size_t col = 0; col < global_size; col++) {
            if (global_mask_matrix[row][col] == STAR) {
                global_col_mask[col] = true;
                covercount++;
            }
        }
    }

    if (covercount >= global_size) {
        return 0;
    }
    return 3;
}

int step3(void) {
    global_item = 0;  // Set the global item before calling find_uncovered_in_matrix
    if (find_uncovered_in_matrix(global_saverow, global_savecol)) {
        global_mask_matrix[global_saverow][global_savecol] = PRIME;
    } else {
        return 5;
    }

    for (size_t ncol = 0; ncol < global_size; ncol++) {
        if (global_mask_matrix[global_saverow][ncol] == STAR) {
            global_row_mask[global_saverow] = true;
            global_col_mask[ncol] = false;
            return 3;
        }
    }

    return 4;
}

int step4(void) {
    // Find alternating sequence of starred and primed zeros
    size_t row = global_saverow;
    size_t col = global_savecol;
    bool found_star = false;
    bool found_prime = false;

    // Find starred zero in column
    for (size_t i = 0; i < global_size; i++) {
        if (global_mask_matrix[i][col] == STAR) {
            row = i;
            found_star = true;
            break;
        }
    }

    if (found_star) {
        // Find primed zero in row
        for (size_t j = 0; j < global_size; j++) {
            if (global_mask_matrix[row][j] == PRIME) {
                col = j;
                found_prime = true;
                break;
            }
        }
    }

    // Update masks
    if (found_star && found_prime) {
        global_mask_matrix[global_saverow][global_savecol] = STAR;
        global_mask_matrix[row][col] = NORMAL;
    }

    // Clear primes and reset masks
    for (size_t i = 0; i < global_size; i++) {
        for (size_t j = 0; j < global_size; j++) {
            if (global_mask_matrix[i][j] == PRIME) {
                global_mask_matrix[i][j] = NORMAL;
            }
        }
    }

    for (size_t i = 0; i < global_size; i++) {
        global_row_mask[i] = false;
        global_col_mask[i] = false;
    }

    return 2;
}

int step5(void) {
    double h = std::numeric_limits<double>::max();
    
    // Find minimum uncovered value
    for (size_t row = 0; row < global_size; row++) {
        if (!global_row_mask[row]) {
            for (size_t col = 0; col < global_size; col++) {
                if (!global_col_mask[col]) {
                    if (h > global_matrix[row][col] && global_matrix[row][col] != 0) {
                        h = global_matrix[row][col];
                    }
                }
            }
        }
    }

    // Add h to covered rows
    for (size_t row = 0; row < global_size; row++) {
        if (global_row_mask[row]) {
            for (size_t col = 0; col < global_size; col++) {
                global_matrix[row][col] += h;
            }
        }
    }

    // Subtract h from uncovered columns
    for (size_t col = 0; col < global_size; col++) {
        if (!global_col_mask[col]) {
            for (size_t row = 0; row < global_size; row++) {
                global_matrix[row][col] -= h;
            }
        }
    }

    return 3;
}

void solve_munkres(double* matrix, size_t rows, size_t columns) {
    global_size = std::max(rows, columns);
    
    // Copy input matrix to global matrix
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < columns; j++) {
            global_matrix[i][j] = matrix[i * columns + j];
        }
    }

    // Initialize masks
    for (size_t i = 0; i < global_size; i++) {
        global_row_mask[i] = false;
        global_col_mask[i] = false;
    }

    // Prepare matrix values
    replace_infinites();
    minimize_along_direction(rows >= columns);
    minimize_along_direction(rows < columns);

    // Follow the steps
    int step = 1;
    while (step) {
        switch (step) {
            case 1:
                step = step1();
                break;
            case 2:
                step = step2();
                break;
            case 3:
                step = step3();
                break;
            case 4:
                step = step4();
                break;
            case 5:
                step = step5();
                break;
        }
    }

    // Store results back to input matrix
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < columns; col++) {
            if (global_mask_matrix[row][col] == STAR) {
                matrix[row * columns + col] = 0;
            } else {
                matrix[row * columns + col] = -1;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input_file> <xclbin>" << std::endl;
        return EXIT_FAILURE;
    }

    try {
        // Initialize FPGA
        std::string binaryFile = argv[2];
        auto fileBuf = xcl::read_binary_file(binaryFile);

        // OpenCL setup
        std::vector<cl::Device> devices = xcl::get_xil_devices();
        if (devices.empty()) {
            std::cerr << "No Xilinx devices found" << std::endl;
            return EXIT_FAILURE;
        }
        
        cl::Device device = devices[0];
        cl::Context context(device);
        cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
        std::string device_name = device.getInfo<CL_DEVICE_NAME>();

        // Create program from binary
        cl::Program::Binaries bins;
        bins.push_back({fileBuf.data(), fileBuf.size()});
        devices.resize(1);
        cl::Program program(context, devices, bins);
        cl::Kernel krnl_find_uncovered(program, "find_uncovered_kernel");

        OCL_CHECK(err, cl::Buffer buffer_matrix(context, CL_MEM_READ_ONLY, sizeof(double)*MAX_SIZE*MAX_SIZE, NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_mask_matrix(context, CL_MEM_READ_ONLY, sizeof(int)*MAX_SIZE*MAX_SIZE, NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_row_mask(context, CL_MEM_READ_ONLY, sizeof(bool), NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_col_mask(context, CL_MEM_READ_ONLY, sizeof(bool), NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_item(context, CL_MEM_READ_ONLY, sizeof(double), NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_size(context, CL_MEM_READ_ONLY, sizeof(size_t), NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_row_out(context, CL_MEM_WRITE_ONLY, sizeof(size_t), NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_col_out(context, CL_MEM_WRITE_ONLY, sizeof(size_t), NULL, &err));


        // Set kernel arguments
        OCL_CHECK(err, err=krnl_find_uncovered.setArg(0, buffer_matrix));
        OCL_CHECK(err, err=krnl_find_uncovered.setArg(1, buffer_mask_matrix));
        OCL_CHECK(err, err=krnl_find_uncovered.setArg(2, buffer_row_mask));
        OCL_CHECK(err, err=krnl_find_uncovered.setArg(3, buffer_col_mask));
        OCL_CHECK(err, err=krnl_find_uncovered.setArg(4, buffer_item));
        OCL_CHECK(err, err=krnl_find_uncovered.setArg(5, buffer_size));
        OCL_CHECK(err, err=krnl_find_uncovered.setArg(6, buffer_row_out));
        OCL_CHECK(err, err=krnl_find_uncovered.setArg(7, buffer_col_out));

        OCL_CHECK(err, global_matrix = (double*)q.enqueueMapBuffer(buffer_matrix, CL_TRUE, CL_MAP_WRITE, 0, sizeof(double)*MAX_SIZE*MAX_SIZE, NULL, NULL, &err));
        OCL_CHECK(err, global_mask_matrix = (double*)q.enqueueMapBuffer(buffer_mask_matrix, CL_TRUE, CL_MAP_WRITE, 0, sizeof(double)*MAX_SIZE*MAX_SIZE, NULL, NULL, &err));
        OCL_CHECK(err, global_row_mask = (bool*)q.enqueueMapBuffer(buffer_row_mask, CL_TRUE, CL_MAP_WRITE, 0, sizeof(bool), NULL, NULL, &err));
        OCL_CHECK(err, global_col_mask = (bool*)q.enqueueMapBuffer(buffer_col_mask, CL_TRUE, CL_MAP_WRITE, 0, sizeof(bool), NULL, NULL, &err));
        OCL_CHECK(err, global_item = (double*)q.enqueueMapBuffer(buffer_item, CL_TRUE, CL_MAP_WRITE, 0, sizeof(double), NULL, NULL, &err));
        OCL_CHECK(err, global_size = (size_t*)q.enqueueMapBuffer(buffer_size, CL_TRUE, CL_MAP_WRITE, 0, sizeof(size_t), NULL, NULL, &err));
        OCL_CHECK(err, global_saverow = (size_t*)q.enqueueMapBuffer(buffer_row_out, CL_TRUE, CL_MAP_READ, 0, sizeof(size_t), NULL, NULL, &err));
        OCL_CHECK(err, global_savecol= (size_t*)q.enqueueMapBuffer(buffer_col_out, CL_TRUE, CL_MAP_READ, 0, sizeof(size_t), NULL, NULL, &err));

        // Read input file
        std::ifstream file(argv[1]);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << argv[1] << std::endl;
            return EXIT_FAILURE;
        }

        std::string temp;
        std::getline(file, temp);
        std::stringstream ss(temp);
        size_t rows, columns;
        ss >> rows >> columns;

        if (rows > MAX_SIZE || columns > MAX_SIZE) {
            std::cerr << "Matrix size exceeds maximum allowed size of " << MAX_SIZE << std::endl;
            return EXIT_FAILURE;
        }

        double matrix[MAX_SIZE][MAX_SIZE];
        for (size_t row = 0; row < rows; row++) {
            std::getline(file, temp);
            std::stringstream ss(temp);
            for (size_t col = 0; col < columns; col++) {
                ss >> matrix[row][col];
            }
        }

        // Display begin matrix state
        std::cout << "\nInitial matrix state:" << std::endl;
        for (size_t row = 0; row < rows; row++) {
            for (size_t col = 0; col < columns; col++) {
                std::cout.width(2);
                std::cout << matrix[row][col] << ",";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Solve the assignment problem
        solve_munkres(matrix, rows, columns, context, q, krnl_find_uncovered);

        // Display solved matrix
        std::cout << "Solved matrix state:" << std::endl;
        for (size_t row = 0; row < rows; row++) {
            for (size_t col = 0; col < columns; col++) {
                std::cout.width(2);
                std::cout << matrix[row][col] << ",";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Verify solution
        for (size_t row = 0; row < rows; row++) {
            int rowcount = 0;
            for (size_t col = 0; col < columns; col++) {
                if (matrix[row][col] == 0)
                    rowcount++;
            }
            if (rowcount != 1)
                std::cerr << "Row " << row << " has " << rowcount << " columns that have been matched." << std::endl;
        }

        for (size_t col = 0; col < columns; col++) {
            int colcount = 0;
            for (size_t row = 0; row < rows; row++) {
                if (matrix[row][col] == 0)
                    colcount++;
            }
            if (colcount != 1)
                std::cerr << "Column " << col << " has " << colcount << " rows that have been matched." << std::endl;
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in main: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
} 
