#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <sys/types.h>
#include <vector>
#include <limits>
#include <tuple>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

inline bool is_line_noninterfering_2d(
    int y0, int x0, int y1, int x1,
    const py::detail::unchecked_reference<int,2>& arr,
    int label_i, int label_j)
{

    int dy = std::abs(y1 - y0);
    int dx = std::abs(x1 - x0);
    int sy = (y1 >= y0) ? 1 : -1;
    int sx = (x1 >= x0) ? 1 : -1;
    int err = dx - dy;
    int y = y0;
    int x = x0;

    while (true)
    {
        int lbl = arr(y,x);
        if (lbl != 0 && lbl != label_i && lbl != label_j)
            return false;

        if (y == y1 && x == x1)
            return true;

        int e2 = 2 * err;
        if (e2 > -dy)
        {
            err -= dy;
            x += sx;
        }
        if (e2 < dx)
        {
            err += dx;
            y += sy;
        }
    }

    return true;
}

inline bool is_line_noninterfering_3d(
int z0, int y0, int x0,
int z1, int y1, int x1,
const int* p,
int strideY,
int strideZ,
int label_i, int label_j)
{

int dz = std::abs(z1 - z0);
int dy = std::abs(y1 - y0);
int dx = std::abs(x1 - x0);

int sx = (dx == 0) ? 0 : (x1 > x0 ? 1 : -1);
int sy = (dy == 0) ? 0 : (y1 > y0 ? 1 : -1);
int sz = (dz == 0) ? 0 : (z1 > z0 ? 1 : -1);

const int* q = p + sz * dz * strideZ + sy * dy * strideY + sx * dx; // Pointer to end point

if (dx >= dy && dx >= dz) {
    if (*(p + sx) == label_i || *(p + sy * strideY + sx) == label_i || *(p + sz * strideZ + sx) == label_i || *(p + sz * strideZ + sy * strideY + sx) == label_i) {
        return false;
    }
    
    if (*(q - sx) == label_j || *(q - sy * strideY - sx) == label_j || *(q - sz * strideZ - sx) == label_j || *(q - sz * strideZ - sy * strideY - sx) == label_j) {
        return false;
    }

    int err_y = 2 * dy - dx;
    int err_z = 2 * dz - dx;

    while (true)
    {
        int lbl = *p;
        if (lbl != 0 && lbl != label_i && lbl != label_j)
            return false;
    
        if (p == q)
            return true;
            
        if (err_z > 0)
        {
            err_z -= 2*dx;
            p += sz * strideZ;
        }
    
        if (err_y > 0)
        {
            err_y -= 2*dx;
            p += sy * strideY;
        }

        p += sx;
        err_y += 2 * dy;
        err_z += 2 * dz;
    }
}

if (dy >= dx && dy >= dz) {
    if (*(p + sy * strideY) == label_i || *(p + sy * strideY + sx) == label_i || *(p + sz * strideZ + sy * strideY) == label_i || *(p + sz * strideZ + sy * strideY + sx) == label_i) {
        return false;
    }
    
    if (*(q - sy * strideY) == label_j || *(q - sy * strideY - sx) == label_j || *(q - sz * strideZ - sy * strideY) == label_j || *(q - sz * strideZ - sy * strideY - sx) == label_j) {
        return false;
    }

    int err_x = 2 * dx - dy;
    int err_z = 2 * dz - dy;

    while (true)
    {
        int lbl = *p;
        if (lbl != 0 && lbl != label_i && lbl != label_j)
            return false;
    
        if (p == q)
            return true;
    
        if (err_z > 0)
        {
            err_z -= 2*dy;
            p += sz * strideZ;
        }
    
        if (err_x > 0)
        {
            err_x -= 2*dy;
            p += sx;
        }
        
        p += sy * strideY;
        err_x += 2 * dx;
        err_z += 2 * dz;
    }
}

if (dz >= dx && dz >= dy) {
    //if (*(p + sz * strideZ) == label_i || *(p + sz * strideZ + sx) == label_i || *(p + sz * strideZ + sy * strideY) == label_i || *(p + sz * strideZ + sy * strideY + sx) == label_i) {
        //return false;
    //}
    
    if (*(q - sz * strideZ) == label_j || *(q - sz * strideZ - sx) == label_j || *(q - sz * strideZ - sy * strideY) == label_j || *(q - sz * strideZ - sy * strideY - sx) == label_j) {
        return false;
    }

    int err_x = 2 * dx - dz;
    int err_y = 2 * dy - dz;

    int lbl = *p;
    while (true)
    {
        lbl = *p;
        if (lbl != 0 && lbl != label_i && lbl != label_j)
            return false;
    
        if (p == q)
            return true;
    
        if (err_y > 0)
        {
            err_y -= 2*dz;
            p += sy * strideY;
        }
    
        if (err_x > 0)
        {
            err_x -= 2*dz;
            p += sx;
        }
        p += sz * strideZ;
        err_x += 2 * dx;
        err_y += 2 * dy;
    }
}

return true;
}

std::tuple<double,int,int,int,int>
find_shortest_noninterfering_2d_serial(
    const py::array_t<int>& borders_i,
    const py::array_t<int>& borders_j,
    const py::array_t<int>& labeled_arr,
    int label_i, int label_j
) {
    auto bi = borders_i.unchecked<2>();
    auto bj = borders_j.unchecked<2>();
    auto arr = labeled_arr.unchecked<2>();

    double best_dist = std::numeric_limits<double>::infinity();
    int best_y0=0, best_x0=0, best_y1=0, best_x1=0;

    for (ssize_t a = 0; a < bi.shape(0); ++a) {
        int y0 = bi(a,0);
        int x0 = bi(a,1);
        for (ssize_t b = 0; b < bj.shape(0); ++b) {
            int y1 = bj(b,0);
            int x1 = bj(b,1);
            double dist = std::abs(y1 - y0) + std::abs(x1 - x0);
            if (dist >= best_dist)
                continue;
            if (is_line_noninterfering_2d(y0, x0, y1, x1, arr, label_i, label_j)) {
                best_dist = dist;
                best_y0 = y0; best_x0 = x0;
                best_y1 = y1; best_x1 = x1;
            }
        }
    }

    if (best_dist < std::numeric_limits<double>::infinity())
        return {best_dist, best_y0, best_x0, best_y1, best_x1};
    else
        return {-1.0, 0,0,0,0};
}

std::tuple<int,int,int>
find_shortest_noninterfering_3d_serial(
    const py::array_t<int>& borders_i,
    const py::array_t<int>& borders_j,
    const py::array_t<int>& center_j,
    const py::array_t<int, py::array::c_style>& labeled_arr,
    int label_i, int label_j,
    double edge_threshold = std::numeric_limits<double>::infinity()
) {
    auto bi = borders_i.unchecked<2>();
    auto bj = borders_j.unchecked<2>();
    auto cj = center_j.unchecked<1>();

    const int* data = labeled_arr.data();
    int strideY = labeled_arr.shape(2);
    int strideZ = labeled_arr.shape(1) * strideY;

    double best_dist = std::numeric_limits<double>::infinity();
    int best_border_idx_i=0, best_border_idx_j=0;

    int max_dist = labeled_arr.shape(0) + labeled_arr.shape(1) + labeled_arr.shape(2);
    std::vector<std::vector<int>> bi_to_center_j_buckets(max_dist + 1);

    for (ssize_t a = 0; a < bi.shape(0); ++a) {
        int z0 = bi(a,0);
        int y0 = bi(a,1);
        int x0 = bi(a,2);

        int dist_to_center_j = std::abs(z0 - cj(0)) + std::abs(y0 - cj(1)) + std::abs(x0 - cj(2));

        bi_to_center_j_buckets[dist_to_center_j].push_back(a);
    }

    for (int dist_to_center_j = 0;  dist_to_center_j < max_dist + 1; dist_to_center_j++) {
        for (int a : bi_to_center_j_buckets[dist_to_center_j]) {
            
            int z0 = bi(a,0);
            int y0 = bi(a,1);
            int x0 = bi(a,2);

            const int* p = data + strideZ * z0 + strideY * y0 + x0;

            for (ssize_t b = 0; b < bj.shape(0); ++b) {
                int z1 = bj(b,0);
                int y1 = bj(b,1);
                int x1 = bj(b,2);
                int dist = std::abs(z1 - z0) + std::abs(y1 - y0) + std::abs(x1 - x0);

                if (dist >= best_dist || dist > edge_threshold) {
                    continue;
                }

                if (is_line_noninterfering_3d(z0, y0, x0, z1, y1, x1, p, strideY, strideZ, label_i, label_j)) {
                    best_dist = dist;
                    best_border_idx_i = a;
                    best_border_idx_j = b;
                }
            }

        }
    }

    if (best_dist < std::numeric_limits<double>::infinity())
        return {best_dist, best_border_idx_i, best_border_idx_j};
    else
        return {-1.0, 0,0};
}

std::tuple<double,int,int,int,int>
find_shortest_noninterfering_2d_parallel(
    const py::array_t<int>& borders_i,
    const py::array_t<int>& borders_j,
    const py::array_t<int>& labeled_arr,
    int label_i, int label_j, int num_threads)
{
    auto bi = borders_i.unchecked<2>();
    auto bj = borders_j.unchecked<2>();
    auto arr = labeled_arr.unchecked<2>();

    size_t ni = bi.shape(0);
    size_t nj = bj.shape(0);
    size_t total_pairs = ni * nj;

    double best_dist = std::numeric_limits<double>::infinity();
    int best_y0=0, best_x0=0, best_y1=0, best_x1=0;

    std::vector<double> best_dist_thread(num_threads, std::numeric_limits<double>::infinity());
    std::vector<std::tuple<int,int,int,int>> best_coords_thread(num_threads, {0,0,0,0});

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (size_t idx = 0; idx < total_pairs; ++idx) {
        int tid = omp_get_thread_num();

        size_t a = idx / nj;
        size_t b = idx % nj;

        int y0 = bi(a,0);
        int x0 = bi(a,1);
        int y1 = bj(b,0);
        int x1 = bj(b,1);

        double dist = std::abs(y1 - y0) + std::abs(x1 - x0);
        if (dist >= best_dist_thread[tid]) continue;

        if (is_line_noninterfering_2d(y0, x0, y1, x1, arr, label_i, label_j)) {
            best_dist_thread[tid] = dist;
            best_coords_thread[tid] = {y0, x0, y1, x1};
        }
    }

    for (int t = 0; t < num_threads; ++t) {
        if (best_dist_thread[t] < best_dist) {
            best_dist = best_dist_thread[t];
            std::tie(best_y0, best_x0, best_y1, best_x1) = best_coords_thread[t];
        }
    }

    if (best_dist < std::numeric_limits<double>::infinity())
        return {best_dist, best_y0, best_x0, best_y1, best_x1};
    else
        return {-1.0, 0,0,0,0};
}

std::tuple<double,int,int,int,int,int,int>
find_shortest_noninterfering_3d_parallel(
    const py::array_t<int>& borders_i,
    const py::array_t<int>& borders_j,
    const py::array_t<int>& labeled_arr,
    int label_i, int label_j, int num_threads)
{
    auto bi = borders_i.unchecked<2>();
    auto bj = borders_j.unchecked<2>();
    auto arr = labeled_arr.unchecked<3>();

    size_t ni = bi.shape(0);
    size_t nj = bj.shape(0);
    size_t total_pairs = ni * nj;

    const int* data = labeled_arr.data();
    int strideY = labeled_arr.shape(2);
    int strideZ = labeled_arr.shape(1) * strideY;

    double best_dist = std::numeric_limits<double>::infinity();
    int best_z0=0, best_y0=0, best_x0=0, best_z1=0, best_y1=0, best_x1=0;

    std::vector<double> best_dist_thread(num_threads, std::numeric_limits<double>::infinity());
    std::vector<std::tuple<int,int,int,int,int,int>> best_coords_thread(num_threads, {0,0,0,0,0,0});

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1000)
    for (size_t idx = 0; idx < total_pairs; ++idx) {
        int tid = omp_get_thread_num();

        size_t a = idx / nj;
        size_t b = idx % nj;

        int z0 = bi(a,0);
        int y0 = bi(a,1);
        int x0 = bi(a,2);

        const int* p = data + strideZ * z0 + strideY * y0 + x0;

        int z1 = bj(b,0);
        int y1 = bj(b,1);
        int x1 = bj(b,2);

        double dist = std::abs(z1 - z0) + std::abs(y1 - y0) + std::abs(x1 - x0);
        if (dist >= best_dist_thread[tid]) continue;

        if (is_line_noninterfering_3d(z0, y0, x0, z1, y1, x1, p, strideY, strideZ, label_i, label_j)) {
            best_dist_thread[tid] = dist;
            best_coords_thread[tid] = {z0, y0, x0, z1, y1, x1};
        }
    }

    for (int t = 0; t < num_threads; ++t) {
        if (best_dist_thread[t] < best_dist) {
            best_dist = best_dist_thread[t];
            std::tie(best_z0, best_y0, best_x0, best_z1, best_y1, best_x1) = best_coords_thread[t];
        }
    }

    if (best_dist < std::numeric_limits<double>::infinity())
        return {best_dist, best_z0, best_y0, best_x0, best_z1, best_y1, best_x1};
    else
        return {-1.0,0,0,0,0,0,0};
}

std::vector<std::tuple<int,int,int,int,int>>
compute_edge_information_parallel(
    const std::vector<py::array_t<int>>& centers,
    const std::vector<py::array_t<int>>& borders,
    const py::array_t<int, py::array::c_style>& labeled,
    const std::vector<int>& region_labels,
    const std::vector<double>& particle_radii,
    int edge_threshold,
    int num_threads)
{
    int n = centers.size();

    std::vector<std::pair<int,int>> pairs;
    pairs.reserve(n*(n-1)/2);

    for(int i=0;i<n;i++)
    {
        for(int j=i+1;j<n;j++)
        {
            pairs.emplace_back(i,j);
        }
    }

    size_t pair_count = pairs.size();

    std::vector<std::tuple<int,int,int,int,int>> edges;
    edges.reserve(pair_count / 4);

    #pragma omp parallel num_threads(num_threads)
    {
        std::vector<std::tuple<int,int,int,int,int>> local_edges;
        local_edges.reserve(pair_count / 4 / num_threads);

        #pragma omp for schedule(dynamic, 1024)
        for (size_t idx = 0; idx < pair_count; idx++)
        {
            const auto& [i,j] = pairs[idx];

            auto ci = centers[i].unchecked<1>();
            auto cj = centers[j].unchecked<1>();
            int zc = ci(0) - cj(0);
            int yc = ci(1) - cj(1);
            int xc = ci(2) - cj(2);

            double center_dist = std::abs(zc) + std::abs(yc) + std::abs(xc);
            //std::cout << "Processing pair: " << i << "," << j << std::endl;

            if (center_dist > edge_threshold + particle_radii[i] + particle_radii[j])
                continue;

            auto result = find_shortest_noninterfering_3d_serial(
                borders[i], borders[j],
                centers[j],
                labeled,
                region_labels[i], region_labels[j],
                edge_threshold
            );

            int dist, border_idx_i, border_idx_j;

            std::tie(dist,border_idx_i, border_idx_j) = result;

            if (dist >= 0 && dist <= edge_threshold)
            {
                local_edges.emplace_back(
                    i, j, dist, border_idx_i, border_idx_j
                );
            }
        }

        #pragma omp critical
        edges.insert(edges.end(), local_edges.begin(), local_edges.end());
    }

    return edges;
}

PYBIND11_MODULE(line_interference_cpp, m) {
    m.def("find_shortest_noninterfering_2d_serial", &find_shortest_noninterfering_2d_serial,
          "Find shortest non-interfering line between two (two-dimensional) border sets",
          py::arg("borders_i"), py::arg("borders_j"),
          py::arg("labeled_arr"), py::arg("label_i"), py::arg("label_j"));

    m.def("find_shortest_noninterfering_2d_parallel", &find_shortest_noninterfering_2d_parallel,
          "Find shortest non-interfering line between two (two-dimensional) border sets (using omp)",
          py::arg("borders_i"), py::arg("borders_j"),
          py::arg("labeled_arr"), py::arg("label_i"), py::arg("label_j"),
          py::arg("num_threads"));

    m.def("find_shortest_noninterfering_3d_serial", &find_shortest_noninterfering_3d_serial,
          "Find shortest non-interfering line between two (three-dimensional) border sets",
          py::arg("borders_i"), py::arg("borders_j"), py::arg("center_j"), py::arg("labeled_arr"), py::arg("label_i"),
          py::arg("label_j"), py::arg("edge_threshold") = std::numeric_limits<double>::infinity());

    m.def("find_shortest_noninterfering_3d_parallel", &find_shortest_noninterfering_3d_parallel,
          "Find shortest non-interfering line between two (three-dimensional) border sets (using omp)",
          py::arg("borders_i"), py::arg("borders_j"),
          py::arg("labeled_arr"), py::arg("label_i"), py::arg("label_j"),
          py::arg("num_threads"));
    m.def(
        "compute_edge_information_parallel",
        &compute_edge_information_parallel,
        R"(
        Compute the thresholded particle graph in parallel.

        Parameters
        ----------

        centers : list[numpy.ndarray]
            Particle centroids stored in 1D Numpy arrays. Used in a pruning criterion to avoid full checks for particles, which can never have an edge smaller than the threshold.
        borders : list[numpy.ndarray]
            Boundary coordinates of each particle stored in 2D Numpy arrays.
        labeled : numpy.ndarray
            Labeled input image.
        region_labels : list[int]
            Labels corresponding to the particles.
        particle_radii : list[float]
            Radii of the particles. Used in the same pruning criterion as centers.
        edge_threshold : double
            Threshold on the edge length in the particle graph.
        num_threads : int
            Number of threads used in parallel.

        Returns
        -------

        list[tuple[int, int, float, int, int]]
            Each tuple represents one edge and contains

            (i, j, dist, border_index_i, border_index_j)

            where
            - i, j : int
                Indices of the edge.
            - dist : float
                Distance or weight of the edge.
            - border_index_i, border_index_j : int
                Indices into borders[i] and borders[j] at which the minimal unobstructed line is obtained.
        )",
        py::arg("centers"),
        py::arg("borders"),
        py::arg("labeled"),
        py::arg("region_labels"),
        py::arg("particle_radii"),
        py::arg("edge_threshold"),
        py::arg("num_threads")
    );
}