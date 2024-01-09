#include <queue>
#include <limits>
#include <cmath>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <stdio.h>


const float INF = std::numeric_limits<float>::infinity();

// represents a single pixel
class Node {
  public:
    int idx; // index in the flattened grid
    float cost; // cost of traversing this pixel
    int path_length; // the length of the path to reach this node

    Node(int i, float c, int path_length) : idx(i), cost(c), path_length(path_length) {}
};

// the top of the priority queue is the greatest element by default,
// but we want the smallest, so flip the sign
bool operator<(const Node &n1, const Node &n2) {
  return n1.cost > n2.cost;
}

// See for various grid heuristics:
// http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#S7
// L_\inf norm (diagonal distance)
inline float linf_norm(int i0, int j0, int i1, int j1) {
  return std::max(std::abs(i0 - i1), std::abs(j0 - j1));
}

// L_1 norm (manhattan distance)
inline float l1_norm(int i0, int j0, int i1, int j1) {
  return std::abs(i0 - i1) + std::abs(j0 - j1);
}

inline float euclidean_distance(int i0, int j0, int i1, int j1) {
    float xd =(float)(i0 - i1);
    float yd =(float)(j0 - j1);
    float dist = std::sqrt(xd*xd+yd*yd);
  return dist;
}

// weights:        flattened h x w grid of costs
// h, w:           height and width of grid
// start, goal:    index of start/goal in flattened grid
// diag_ok:        if true, allows diagonal moves (8-conn.)
// paths (output): for each node, stores previous node in path
//extern "C" bool astar(
static PyObject *astar(PyObject *self, PyObject *args) {
  const PyArrayObject* weights_object;
  int h;
  int w;
  int start;
  int goal;
  int diag_ok;

  if (!PyArg_ParseTuple(
        args, "Oiiiii", // i = int, O = object
        &weights_object,
        &h, &w,
        &start, &goal,
        &diag_ok))
    return NULL;

  float* weights = (float*) weights_object->data;
  int* paths = new int[h * w];
  int* contours = new int[h *w];
  for (int i = 0; i < h * w; ++i)
    contours[i] = -1;
  int path_length = -1;
  int contour_length = -1;

  Node start_node(start, 0., 1);

  float* costs = new float[h * w];
  for (int i = 0; i < h * w; ++i)
    costs[i] = INF;
  costs[start] = 0.;

  std::priority_queue<Node> nodes_to_visit;
  nodes_to_visit.push(start_node);

  int* nbrs = new int[8];

  while (!nodes_to_visit.empty()) {
    // .top() doesn't actually remove the node
    Node cur = nodes_to_visit.top();

//    if (cur.idx == goal) {
//      path_length = cur.path_length;
//      break;
//    }

    nodes_to_visit.pop();

    int row = cur.idx / w;
    int col = cur.idx % w;
    // check bounds and find up to eight neighbors: top to bottom, left to right
    nbrs[4] = (diag_ok && row > 0 && col > 0)          ? cur.idx - w - 1   : -1;
    nbrs[0] = (row > 0)                                ? cur.idx - w       : -1;
    nbrs[5] = (diag_ok && row > 0 && col + 1 < w)      ? cur.idx - w + 1   : -1;
    nbrs[1] = (col > 0)                                ? cur.idx - 1       : -1;
    nbrs[2] = (col + 1 < w)                            ? cur.idx + 1       : -1;
    nbrs[6] = (diag_ok && row + 1 < h && col > 0)      ? cur.idx + w - 1   : -1;
    nbrs[3] = (row + 1 < h)                            ? cur.idx + w       : -1;
    nbrs[7] = (diag_ok && row + 1 < h && col + 1 < w ) ? cur.idx + w + 1   : -1;
    bool contour_append = false;
    float heuristic_cost;
    for (int i = 0; i < 8; ++i) {
      if (nbrs[i] >= 0) {
        // the sum of the cost so far and the cost of this move
        float weight_action;
        if (i <= 3) {
            weight_action = weights[nbrs[i]];
        }
        else{
            weight_action = 1.414 * weights[nbrs[i]];
        }
        float new_cost = costs[cur.idx] + weight_action;
        if (new_cost < costs[nbrs[i]]) {
          // paths with lower expected cost are explored first
          // float priority = new_cost + heuristic_cost;
          float priority = new_cost;
          nodes_to_visit.push(Node(nbrs[i], priority, cur.path_length + 1));

          costs[nbrs[i]] = new_cost;
          paths[nbrs[i]] = cur.idx;
        }
        if (new_cost > 9000 && contour_append == false){
            contours[cur.idx] = cur.idx;
            contour_append = true;
            contour_length = contour_length + 1;
        }
      }
    }
  }

  PyObject *return_val;
  //printf("%d     ",contour_length);
  if (contour_length >= 0) {
    //contour_length = 100;
    int buffer_id = contour_length - 1;

    int buffer[contour_length][3];
    for (int idx = 0; idx < h * w; ++idx){
        if(contours[idx] != -1){
            buffer[buffer_id][0] = idx / w;
            buffer[buffer_id][1] = idx % w;
            buffer[buffer_id][2] = costs[idx];
            //printf("i: %d,index: %d, cidx:%d %d %d\n", buffer_id, idx, contours[idx], buffer[buffer_id][0], buffer[buffer_id][1]);
            buffer_id--;
            if (buffer_id == 0){
                break;
            }
        }
    }
    npy_intp dims[2] = {contour_length - buffer_id-1, 3};
    PyArrayObject* contour = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT32);
    npy_int32 *iptr, *jptr, *kptr;
    for (npy_intp i = dims[0] - 1; i >=0; --i) {
        //printf("%d ",path->data);
        iptr = (npy_int32*) (contour->data + i * contour->strides[0]);
        jptr = (npy_int32*) (contour->data + i * contour->strides[0] + contour->strides[1]);
        kptr = (npy_int32*) (contour->data + i * contour->strides[0] + contour->strides[1] + contour->strides[1]);

        *iptr = buffer[i+buffer_id+1][0];
        *jptr = buffer[i+buffer_id+1][1];
        *kptr = buffer[i+buffer_id+1][2];
        // printf("num: %d %d %d\n", i, buffer[i+buffer_id+1][0], buffer[i+buffer_id+1][1]);
    }

    return_val = PyArray_Return(contour);
  }
  else {
    return_val = Py_BuildValue(""); // no soln --> return None
  }

  delete[] costs;
  delete[] nbrs;
  delete[] paths;
  delete[] contours;

  return return_val;
}

static PyMethodDef astar_methods[] = {
    {"astar", (PyCFunction)astar, METH_VARARGS, "astar"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef astar_module = {
    PyModuleDef_HEAD_INIT,"astar", NULL, -1, astar_methods
};

PyMODINIT_FUNC PyInit_astar(void) {
  import_array();
  return PyModule_Create(&astar_module);
}
