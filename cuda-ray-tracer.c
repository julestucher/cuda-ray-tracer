//
//  ray-tracer2.c
//
//
//  Created by Julia Tucher on 11/22/19.
//

#include <stdio.h>
#include <jpeglib.h>
#include <math.h>
#include <stdlib.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>

// define constants to be used

// Quality of output JPEG image
#define OUT_QUALITY 95

// Number of components in the image
#define NUM_COMPONENTS 3

// Define height and width of test image
#define IMAGE_WIDTH 800
#define IMAGE_HEIGHT 500

// Define constants for sizes in host
#define VECTOR_SIZE 3
#define RAY_SIZE 6
#define BSDF_SIZE 8
#define TRIANGLE_SIZE 26
#define LIGHT_SIZE 6

// Define constants for size and offset in kernel
__global__ __constant__ int vectorSize = 3;
__device__ __constant__ int raySize = 6;
__device__ __constant__ int BSDFSize = 8;
__device__ __constant__ int triangleSize = 26;
__device__ __constant__ int lightSize = 6;

/* Create all of the typedefs to be used (classes ported from C++) */

/* The "frame structure" structure contains an image frame (in RGB or grayscale
 * formats) for passing around the CS338 projects.
 */
typedef struct frame_struct {
  JSAMPLE *image_buffer;        /* Points to large array of R,G,B-order/grayscale data
				 * Access directly with:
				 *   image_buffer[num_components*pixel + component]
				 */
  JSAMPLE **row_pointers;       /* Points to an array of pointers to the beginning
				 * of each row in the image buffer.  Use to access
				 * the image buffer in a row-wise fashion, with:
				 *   row_pointers[row][num_components*pixel + component]
				 */
  int image_height;             /* Number of rows in image */
  int image_width;              /* Number of columns in image */
  int num_components;   /* Number of components (usually RGB=3 or gray=1) */
} frame_struct_t;
typedef frame_struct_t *frame_ptr;

// 3D vector - [0] is x coord, [1] is y coord, [2] is z coord
typedef float *Vector3;

// RGB color value - [0] is r coord, [1] is g coord, [2] is b coord
typedef float *Color3;

// measure of light radiance
typedef Color3 Radiance3;

// measure of power of light source
typedef Color3 Power3;

// each ray has origin point and direction (which includes length)
typedef struct Ray_t{
  float origin[3];
  float direction[3];
}Ray_t;
typedef Ray_t *ray_ptr;

// Image struct holds image dimensions and pixel data in Radiance3 format (RGB for light)
typedef struct Image_t{
  int width;
  int height;
  float **data; // float ** with each pixel indexed as [i*width + j] being an RGB(A) value
}Image_t;
typedef Image_t *image_ptr;

// BSDF data... add a comment when you understand what this is !!
typedef struct BSDF_t{
  int glossy; // int (boolean) that is 1 if the BSDF is glossy and 0 if not (if NOT glossy, only has k_L vals)
  float k_L[3];
  float k_G[3];
  float s;
}BSDF_t;
typedef BSDF_t *bsdf_ptr;

// Triangle struct holds info for trianges in a scene
typedef struct Triangle_t{
  float vertex[3][3]; // of size 3, thus a 3x3 array
  float normal[3][3]; // of size 3, another 3x3 array
  bsdf_ptr bsdf;
}Triangle_t;
typedef Triangle_t *triangle_ptr;

// declare the Light source as a struct that has power (over whole sphere) and position
typedef struct Light_t{
  float position[3];
  float power[3];
}Light_t;
typedef Light_t *light_ptr;

// describe each Scene that is being depicted as a set of Triangles and Light sources
typedef struct Scene_t {
  triangle_ptr triangleArray;
  light_ptr lightArray;
  long triangleCount;
  long lightCount;
}Scene_t;
typedef Scene_t *scene_ptr;

// Use a struct to contain info about the Eye that is perceiving the image, namely where it is relative to the scene (including default values in method);
struct Eye_d {
  float zNear;
  float zFar;
  float fieldOfViewX;

} Eye_default = {.zNear = -0.1f, .zFar = -100.0f, .fieldOfViewX = M_PI/2.0f};

typedef struct Eye_d *eye_ptr;


// HELPER METHOD FOR VECTOR COMPUTATION

/* dot: helper method for computing a dot product of two vectors
 * args: two Vector3 to compute dot product of
 * out: float – dot prod
 */
__device__ float dot(Vector3 v1, Vector3 v2){
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

/* magnitude_f: helper method for computing magnitude of a vector given three components
 */
__global__ float magnitude_f(float x, float y, float z){
  return pow((pow(x,2.0f) + pow(y,2.0f) + pow(z,2.0f)), 0.5);
}


/* evaluateFiniteScatteringDensity() takes in a BSDF object and in/out vectors to compute reflectance value
 * and implementation depends of the type of scattering used (Lambertian, etc)
 * (also what about including any other calcs in here? take in n as argument too?)
 * in: bsdf_ptr and w_i, w_o as Vector3 objects
 * out: Color3
 */
__device__ Color3 evaluateFiniteScatteringDensity(bsdf_ptr bsdf, Vector3 w_i, Vector3 w_o, Vector3 n){

  // MAKE SURE THIS IS FREED LATER
  Color3 temp = (Vector3)calloc(3, sizeof(float));

  if(bsdf->glossy){
    Vector3 w_h = (Vector3)calloc(3, sizeof(float));
    w_h[0] = w_i[0] + w_o[0];
    w_h[1] = w_i[1] + w_o[1];
    w_h[2] = w_i[2] + w_o[2];

    float mag = magnitude_f(w_h[0], w_h[1], w_h[2]);
    for(int i = 0; i < 3; i++){
      w_h[i] /= mag;
    }

    float max = dot(w_h, n);
    if (max < 0){ max = 0.0f; }

    // use Blinn-Phong BSDF Scattering density for Glossy Scattering
    for(int i = 0; i < 3; i++){
      temp[i] = (bsdf->k_L[i] + bsdf->k_G[i] * ((bsdf->s + 8.0f) * pow(max, bsdf->s)) / 8.0f) / M_PI;
    }
    free(w_h);
  }

  else{
    /*use Lambertian Scattering method, which returns a constant for each RGB */
    for(int i = 0; i < 3; i++){
      temp[i] = bsdf->k_L[i] / M_PI;
    }
  }

  return temp;
}




/* intersect: calculate the intersection of a Ray and a Triangle
 * arguments: ray pointer, triangle pointer, weights array (float) pointer
 * out: a float representing distance
 */
__device__ float intersect(ray_ptr R, triangle_ptr T, float *weight){

  // declare floats
  float a, dist, epsilon, epsilon2;

  // allocate memory for 2 triangle vectors (2 sides that define Tri), q/r/s vectors
  Vector3 e1 = (Vector3)malloc(3 * sizeof(float)); //later optimization: e1 & e2 are constant for a vertex
  Vector3 e2 = (Vector3)malloc(3 * sizeof(float));
  Vector3 s = (Vector3)malloc(3 * sizeof(float));
  Vector3 r = (Vector3)malloc(3 * sizeof(float));
  Vector3 q = (Vector3)malloc(3 * sizeof(float));

  // define e1 and e2 as difference in vertices, s as difference between R origin point and vertex 0 of T
  for(int i = 0; i < 3; i++){
    e1[i] = T->vertex[1][i] - T->vertex[0][i];
    e2[i] = T->vertex[2][i] - T->vertex[0][i];

    // s is the from V_0 to R origin point
    s[i] = R->origin[i] - T->vertex[0][i];
  }

  // allocate mem for q and define as cross prod of R.direction and diff bn V_0 and V_2
  // q is perpendicular to the plane created by R and e2
  q[0] = R->direction[1] * e2[2] - R->direction[2] * e2[1];
  q[1] = R->direction[2] * e2[0] - R->direction[0] * e2[2];
  q[2] = R->direction[0] * e2[1] - R->direction[1] * e2[0];


  // dot product of e1 and q: are they perpendicular (is R parallel to Triangle?)
  a = dot(e1, q);

  // define r as cross product of s and e1
  // r is perp to plane created by s and e1
  r[0] = s[1] * e1[2] - s[2] * e1[1];
  r[1] = s[2] * e1[0] - s[0] * e1[2];
  r[2] = s[0] * e1[1] - s[1] * e1[0];

  // Barycentric vertex weights
  weight[1] = dot(s, q) / a; // reps angle between q and dist(R, V_0)
  weight[2] = dot(R->direction, r) / a; // reps angle between
  weight[0] = 1.0f - (weight[1] + weight[2]);

  dist = dot(e2, r) / a;

  epsilon = 1e-7f;
  epsilon2 = -1e-10f;


  // changed from fabsf(a)
  int bool1 = (fabsf(a) <= epsilon) || (weight[0] < (-1)*epsilon2) || weight[1] < (-1)*epsilon2 || weight[2] < (-1)*epsilon2 || dist <= 0.0f;

  if(bool1){
    // The ray is nearly parallel to the triangle, or the
    // intersection lies outside the triangle or behind
    // the ray origin: "infinite" distance until intersection.
    dist = INFINITY;
  }

  free(e1);
  free(e2);
  free(q);
  free(r);
  free(s);

  return dist;

}

/* visible() cycles through all of the other triangles to determine whether or not the given point P is visible from the pixel
 * in: point P, vector direction, float that is the distant from the light, pointer to scene
 * out: int representing the boolean value of whether or not P is visible
 */
__device__ int visible(Point3 P, Vector3 direction, float distance, scene_ptr scene) {
  float rayBumpEpsilon = 1e-2;

  // allocate mem and init shadow ray based on P and direction
  ray_ptr shadowRay = (ray_ptr)malloc(sizeof(Ray_t));
  shadowRay->origin[0] = P[0] + (direction[0] * rayBumpEpsilon);
  shadowRay->origin[1] = P[1] + (direction[1] * rayBumpEpsilon);
  shadowRay->origin[2] = P[2] + (direction[2] * rayBumpEpsilon);

  shadowRay->direction[0] = direction[0];
  shadowRay->direction[1] = direction[1];
  shadowRay->direction[2] = direction[2];

  distance -= rayBumpEpsilon;

  // Test each potential shadow caster to see if it lies between P and the light
  float *ignore = (float *)malloc(3 * sizeof(float));
  for(unsigned int s = 0; s < scene->triangleCount; ++s){
    if (intersect(shadowRay, &scene->triangleArray[s], ignore) < distance){ // this triangle array bit is going to need to communicate with device memory, so pass triArray and triCount
      free(ignore);
      return 0;
    }
  }

  free(ignore);
  return 1;

}

/* Implemenation decision of rayTrace: instead of creating several, smaller helper functions that all
 * use the same scope of variables (arguments of rayTrace) and having to copy them into
 * a new scope at each call, implement computeEyeRay, sampleRayTriangle, and shade()
 * within the rayTrace function call
 */

/* rayTraceKernel: trace one ray for given pixel based on thread, given eye viewpoint and triangles
 * Args: Image image, Scene scene, Eye eye, x0/x1/y0/y1 as start and end pixels
 */
__global__ void rayTrace(image_ptr image, scene_ptr scene, eye_ptr eye, float aspect, int x1, int y1){

  printf("enter rayTrace\n");

  // declare scope variables for tracing
  float aspect, s, distance, d, magnitude, distToLight, max;
  triangle_ptr T;
  light_ptr light;

  // allocate memory for Ray
  ray_ptr R = (ray_ptr)calloc(1, sizeof(Ray_t));

  // allocate mem for vector weights
  float *weight = (float *)malloc(3 * sizeof(float));

  //Allocate memory for future intersection point, normal vector, and w_o
  Point3 P = (Point3)malloc(3 * sizeof(float));
  Vector3 n = (Vector3)malloc(3 * sizeof(float));
  Vector3 w_o = (Vector3)malloc(3 * sizeof(float));

  // Allocate memory for shade() variables
  Radiance3 L_o = (Radiance3)calloc(3, sizeof(float)); // light output per triangle
  Radiance3 L_i = (Radiance3)calloc(3, sizeof(float)); // light output per light source
  Vector3 w_i = (Vector3)malloc(3 * sizeof(float));

  // Compute the side of a square at z = -1 based on our
  // horizontal left-edge-to-right-edge field of view
  s = -2.0f * tan(eye->fieldOfViewX * 0.5f);

  // Iterate through each pixel in the range
  for(int i = y0; i < y1; i++){

    for(int j = x0; j < x1; j++){

      /* computeEyeRay function code within this method */
      R->origin[0] = (((j + 0.5f)/x1) - 0.5f) * s * eye->zNear;
      R->origin[1] = (-1)*((i + 0.5f)/y1 - 0.5f) * s * aspect * eye->zNear;
      R->origin[2] = eye->zNear;

      //magnitude = pow((pow(R->origin[0],2.0f) + pow(R->origin[1],2.0f) + pow(R->origin[2],2.0f)), 0.5);
      magnitude = magnitude_f(R->origin[0], R->origin[1], R->origin[2]);

      R->direction[0] = R->origin[0]/magnitude;
      R->direction[1] = R->origin[1]/magnitude;
      R->direction[2] = R->origin[2]/magnitude;

      /* end computeEyeRay function */

      // set the distance to closest known intersection
      distance = INFINITY;

      /* initial image testing code */
      /* image->data[i*x1 + j][0] = (R->direction[0] + 1.0f)/5;
             image->data[i*x1 + j][1] = (R->direction[1] + 1.0f)/5;
             image->data[i*x1 + j][2] = (R->direction[2] + 1.0f)/5;
      */


      // set background color
      L_o[0] = 0.0022f;
      L_o[1] = 0.0032f;
      L_o[2] = 0.0053f;

      // Iterate through each triangle
      for(unsigned int t = 0; t < scene->triangleCount; t++){

	// printf("t %d, i %d, j %d\n", t, i, j);

	//take address of triangle in array
	T = &scene->triangleArray[t];

	// sampleRayTriangle function code within rayTrace method
	d = intersect(R, T, weight);

	// future: best way to minimize branching since shade(..) is expensive
	if(d < distance){



	  //  printf("t %d, i %d, j %d, d %f\n", t, i, j, d);

	  // reset dist because d is closer
	  distance = d;

	  //printf("i %d, j %d, dist %f\n", i , j, distance);

	  // calculate intersection point
	  P[0] = R->origin[0] + (R->direction[0] * d);
	  P[1] = R->origin[1] + (R->direction[1] * d);
	  P[2] = R->origin[2] + (R->direction[2] * d);

	  // Find the interpolated vertex normal at the intersection - aka weighted sum of normal vectors
	  n[0] = T->normal[0][0] * weight[0] + T->normal[1][0] * weight[1] + T->normal[2][0] * weight[2];
	  n[1] = T->normal[0][1] * weight[0] + T->normal[1][1] * weight[1] + T->normal[2][1] * weight[2];
	  n[2] = T->normal[0][2] * weight[0] + T->normal[1][2] * weight[1] + T->normal[2][2] * weight[2];

	  magnitude = magnitude_f(n[0], n[1], n[2]);
	  n[0] /= magnitude;
	  n[1] /= magnitude;
	  n[2] /= magnitude;


	  // Find the opposite direction of R
	  w_o[0] = (-1) * R->direction[0];
	  w_o[1] = (-1) * R->direction[1];
	  w_o[2] = (-1) * R->direction[2];

	  // shade() implementation {shade(scene, T, P, n, w_o, radiance)};

	  // init L_o to black
	  L_o[0] = 0.0f;
	  L_o[1] = 0.0f;
	  L_o[2] = 0.0f;

	  // Iterate through light sources in the scene
	  for(unsigned int k = 0; k < scene->lightCount; ++k){
	    light = &scene->lightArray[k];

	    // init w_i as unit vector version of offset between light position and P
	    distToLight = magnitude_f(light->position[0] - P[0], light->position[1] - P[1], light->position[2] - P[2]);
	    w_i[0] = (light->position[0] - P[0]) / distToLight;
	    w_i[1] = (light->position[1] - P[1]) / distToLight;
	    w_i[2] = (light->position[2] - P[2]) / distToLight;

	    if (visible(P, w_i, distToLight, scene)){
	      L_i[0] = light->power[0] / (4 * M_PI * pow(distToLight, 2.0f));
	      L_i[1] = light->power[1] / (4 * M_PI * pow(distToLight, 2.0f));
	      L_i[2] = light->power[2] / (4 * M_PI * pow(distToLight, 2.0f));

	      // Scatter the light
	      Color3 L_d = evaluateFiniteScatteringDensity(T->bsdf, w_i, w_o, n);

	      max = dot(w_i, n);
	      if(max < 0) { max = 0.0f; }

	      L_o[0] += L_i[0] * L_d[0] * max;
	      L_o[1] += L_i[1] * L_d[1] * max;
	      L_o[2] += L_i[2] * L_d[2] * max;

	      free(L_d); // allocated in helper method evaluateFiniteScatteringDensity

	    }


	  }

	  // end shade implementation

	  //Debugging barycentric
	  //radiance = Radiance3(weight[0], weight[1], weight[2])/15;
	  /*image->data[i*x1 + j][0] = weight[0];
                     image->data[i*x1 + j][1] = weight[1];
                     image->data[i*x1 + j][2] = weight[2];
                     */

	}
	// end sampleRayTriangle

	image->data[i*x1 + j][0] = L_o[0];
	image->data[i*x1 + j][1] = L_o[1];
	image->data[i*x1 + j][2] = L_o[2];

      }

    }
  }

  // free allocated memory (in reverse order)
  printf("about to free\n");
  free(w_i);
  free(L_i);
  free(L_o);
  free(w_o);
  free(n);
  free(P);
  free(weight);
  free(R);
  printf("freed\n");

}

__global__ void rayTraceKernel(float *triangleArray, float *lightArray, float *eye, int riangleCount, int lightCount, float aspect, int width, int height){

}

/* runKernel: This sets up GPU device by allocating the required memory and then calls the kernel on GPU.
 */
void runKernel(image_ptr image, scene_ptr scene, eye_ptr eye, int blockSize){

  // convert the given scene into a 1D array of triangles (vertices, normal vectors, and bsdfs) and light source
  float *triangles_1d, *light_1d, *eye_1d, *image_1d;

  // set size of memory needed for triangle and light arrays
  int triMemSize = scene->triangleCount * TRIANGLE_SIZE * sizeof(float);
  int lightMemSize = scene->lightCount * LIGHT_SIZE * sizeof(float);
  int eyeMemSize = VECTOR_SIZE * sizeof(float);
  int imageMemSize = image->height * image->width * NUM_COMPONENTS * sizeof(float);

  // allocate 1D array memory for triangle array and light source on host
  triangles_1d = (float *)malloc(triMemScene);
  light_1d = (float *)malloc(lightMemSize);
  eye_1d = (float *)malloc(eyeMemSize);
  image_1d = (float *)malloc(imageMemSize);

  // for each triangle in the array, store as consecutive sets of 3 vertices, 3 normal vectors, and 1 BSDF (26 floats total)
  for(int i = 0; i < scene->triangleCount, i++){

    // for the BSDF value, store each part of BSDF (float glossy, float k_L[3], float k_G[3], float s)
    triangles_1d[i * TRIANGLE_SIZE] = scene->triangleArray[i].bsdf->glossy;

    //iterate and store each of the components of the BSDF k_L vector
    for(int k = 0; k < 3; k++){
      triangles_1d[(i * TRIANGLE_SIZE) + 1 + k] = scene->triangleArray[i].bsdf->k_L[k];
    }

    //iterate and store each of the components of the BSDF k_G vector
    for(int k = 0; k < 3; k++){
      triangles_1d[(i * TRIANGLE_SIZE) + 1 + VECTOR_SIZE + k] = scene->triangleArray[i].bsdf->k_G[k];
    }

    //store float s from BSDF
    triangles_1d[(i * TRIANGLE_SIZE) + (2 * VECTOR_SIZE) + 1] = scene->triangleArray[i].bsdf->s;

    // interate through the three vertices in the triangle
    for(int j = 0; j < 3 * VECTOR_SIZE; j++){

      //iterate and store each of the points in each vertex
      for(int k = 0; k < 3; k++){
	triangles_1d[(i * TRIANGLE_SIZE) + BSDF_SIZE + (j * VECTOR_SIZE) + k] = scene->triangleArray[i].vertex[j][k];
      }
    }

    // interate through the three normal vectors in the triangle
    for(int j = 0; j < 3 * VECTOR_SIZE; j++){

      //iterate and store each of the components in each vertex
      for(int k = 0; k < 3; k++){
	triangles_1d[(i * TRIANGLE_SIZE) + BSDF_SIZE + (3 * VECTOR_SIZE) + (j * VECTOR_SIZE) + k] = scene->triangleArray[i].normal[j][k];
      }
    }
  }

  // for each light in the array of light sources, store as sets of two vectors (1 for position, 1 for power)
  for(int i = 0; i < scene->lightCount; i++){

    // iterate through and store the components of position vector
    for(int j = 0; j < VECTOR_SIZE; j++){
      light_1d[i * LIGHT_SIZE + j] = s->lightArray[i].position[j];
    }

    // iterate through and store the components of power vector
    for(int j = 0; j < VECTOR_SIZE; j++){
      light_1d[i * LIGHT_SIZE + VECTOR_SIZE + j] = s->lightArray[i].power[j];
    }
  }

  // init 1D array representation for eye
  eye_1d[0] = eye->zNear;
  eye_1d[1] = eye->zFar;
  eye_1d[2] = eye->fieldOfViewX;

  // declare and allocate pointers for  triangle/light arrays, output image, and eye viewpoint on device memory
  float *device_triangles, *device_lights, *device_image, *device_eye;

  // checking for errors, call cudaMalloc for device arrays
  cudaError_t temp_err = cudaMalloc((void **) &device_triangles, triMemSize);
  checkCudaErrors(temp_err);

  temp_err = cudaMalloc((void **) &device_lights, lightMemSize);
  checkCudaErrors(temp_err);

  temp_err = cudaMalloc((void **) &device_image, imageMemSize);
  checkCudaErrors(temp_err);

  temp_err = cudaMalloc((void **) &device_eye, eyeMemSize);
  checkCudaErrors(temp_err);

  // set all values to zero on device memory and check for errors
  temp_err = cudaMemset(device_triangles, 0, triMemSize);
  checkCudaErrors(temp_err);

  temp_err = cudaMemset(device_lights, 0, lightMemSize);
  checkCudaErrors(temp_err);

  temp_err = cudaMemset(device_image, 0, imageMemSize);
  checkCudaErrors(temp_err);

  temp_err = cudaMemset(device_eye, 0, eyeMemSize);
  checkCudaErrors(temp_err);

  // copy 1D triangle/light arrays and eye to device memory and check for errors
  temp_err = cudaMemcpy(device_triangles, triangles_1d, triMemSize, cudaMemcpyHostToDevice);
  checkCudaErrors(temp_err);

  temp_err = cudaMemcpy(device_lights, light_1d, lightMemSize, cudaMemcpyHostToDevice);
  checkCudaErrors(temp_err);

  temp_err = cudaMemcpy(device_eye, eye_1d, lightMemSize, cudaMemcpyHostToDevice);
  checkCudaErrors(temp_err);

  // calculate aspect ratio (true for all pixels)
  float aspect = (float)image->height/image->width;

  /* declare and init start, stop events for time recording */
  float time_ms = 0.0f;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  /* run size/block_size blocks of block_size threads each */
  dim3 DimGrid(image->width/block_size, image->height/block_size, 1);
  if (image->width%block_size) {
    DimGrid.x++;
  }
  if (image->height%block_size) {
    DimGrid.y++;
  }
  dim3 DimBlock(block_size, block_size, 1);

  /* measure event start before invoking kernel */
  checkCudaErrors(cudaEventRecord(start, 0));

  /* invoke kernel using DimGrid grid dimensions, and DimBlock block dimensions */
  rayTraceKernel<<DimGrid, DimBlock>>(device_triangles, device_lights, device_eye, scene->triangleCount, scene->lightCount, aspect, image->width, image->height);

  temp_err = cudaGetLastError();
  checkCudaErrors(temp_err);

  /* measure event end after invoking kernel */
  checkCudaErrors(cudaEventRecord(end, 0));

  /* wait for end to finish, then measure elapsed time */
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_ms, start, end);
  printf("Time elapsed: %f\n", time_ms);

  // Transfer resulting image from device to host
  temp_err =  cudaMemcpy(device_image, image_1d, imageMemSize, cudaMemcpyDeviceToHost);
  checkCudaErrors(temp_err);

  // iterate through the pixels of the the 1d image array and store in the image_ptr passed in argument

  // iterate through the rows of the image
  for(int i = 0; i < image->height; i++){

    // iterate through the columns of the image
    for(int j = 0; j < image->width; j++){

      // iterate through the RGB components
      for(int k = 0; k < NUM_COMPONENTS; k++){
	image[(i * image_width) + j][k] = image_1d[(i * image_width * NUM_COMPONENTS) + (j * NUM_COMPONENTS) + k];
      }
    }
  }

  // Destroy CUDA events
  cudaEventDestroy(start); cudaEventDestroy(end);

  // free all device memory allocated in this function
  cudaFree(device_triangles); cudaFree(device_lights); cudaFree(device_eye); cudaFree(device_image);

  // free all host memory allocated in this function
  free(triangles_1d); free(light_1d); free(eye_1d); free(image_1d);
}


// Helper function to read in input from formatted Triangle file
void makeTriangleScene(scene_ptr s, char **argv){
  printf("makeTriangle\n");
  FILE *stream = fopen(argv[1], "r");

  for(int k = 0; k < s->triangleCount; k++){

    // read in vertices for each triangle
    for(int i = 0; i < 9; i++){
      fscanf(stream, "%f", &s->triangleArray[k].vertex[i/3][i%3]);
    }


    // read in normal vector values
    for(int i = 0; i < 9; i++){
      fscanf(stream, "%f", &s->triangleArray[k].normal[i/3][i%3]);
    }

    // directionalize normal vectors
    for(int i = 0; i < 3; i++){
      float magnitude = magnitude_f(s->triangleArray[k].normal[i][0], s->triangleArray[k].normal[i][1], s->triangleArray[k].normal[i][2]);
      for(int j = 0; j < 3; j++){
	s->triangleArray[k].normal[i][j] /= magnitude;
      }
    }

    fscanf(stream, "%d", &s->triangleArray[k].bsdf->glossy);

    for(int i = 0; i < 3; i++){
      fscanf(stream, "%f", &s->triangleArray[k].bsdf->k_L[i]);
    }

    // read in bsdf values (glossy scheme)
    if(s->triangleArray[k].bsdf->glossy){
      for(int i = 0; i < 3; i++){
	fscanf(stream, "%f", &s->triangleArray[k].bsdf->k_G[i]);
      }

      fscanf(stream, "%f", &s->triangleArray[k].bsdf->s);
    }
  }

  // read in single light source position and power
  for(int i = 0; i < 3; i++){
    fscanf(stream, "%f", &s->lightArray[0].position[i]);
  }
  for(int i = 0; i < 3; i++){
    fscanf(stream, "%f", &s->lightArray[0].power[i]);
  }

  fclose(stream);
}

/* Render function
 * in: argv (for triangle input)
 * out: pointer to rendered image
 */
image_ptr render(char **argv){

  // number of triangles in the file is passed as the last command-line argument
  int triCount = atoi(argv[3]);

  // set test scene, including one Triangle and one Light source
  scene_ptr testScene = (scene_ptr)calloc(1, sizeof(Scene_t));

  // set test triangle array containing one Triangle with hardcoded values for vertex and normal
  triangle_ptr sceneTriangle = (triangle_ptr)calloc(triCount, sizeof(Triangle_t));

  light_ptr lightSource = (light_ptr)calloc(1, sizeof(Light_t));

  testScene->triangleArray = sceneTriangle;
  testScene->lightArray = lightSource;
  testScene->triangleCount = triCount;
  testScene->lightCount = 1;

  for(int i = 0; i < triCount; i++){
    bsdf_ptr bsdf = (bsdf_ptr)calloc(1, sizeof(BSDF_t));
    sceneTriangle[i].bsdf = bsdf;
  }

  makeTriangleScene(testScene, argv);

  // set temp Image
  image_ptr image = (image_ptr)calloc(1, sizeof(Image_t));
  image->width = IMAGE_WIDTH;
  image->height = IMAGE_HEIGHT;
  image->data = (Radiance3 *)calloc(image->width * image->height, sizeof(Radiance3));

  // allocate space for the RGB value of each pixel
  for(int i = 0; i < image->width * image->height; i++){
    image->data[i] = (float *)calloc(3, sizeof(float));
  }

  eye_ptr eye = &Eye_default;

  //
  runKernel(image, testScene, eye, image->width, image->height);

  // run the serial version
  // rayTrace(image, testScene, eye, image->width, image->height);

  for(int i = 0; i < testScene->triCount; i++){
    free(testScene->triangleArray[i].bsdf);
  }
  free(testScene->triangleArray);
  free(testScene->lightArray);
  free(testScene);

  return image;
}

// JPEG file info: Once rasterization is complete, transform data from Image struct format into JPEG file and write to file based on command line argument
/*
 * allocate/destroy_frame allocate a frame_struct_t and fill in the
 *  blanks appropriately (including allocating the actual frames), and
 *  then destroy them afterwards.
 */
frame_ptr allocate_frame(int height, int width, int num_components){


  printf("allocate frame\n");
  int row_stride;               /* physical row width in output buffer */
  int i;
  frame_ptr p_info;             /* Output frame information */

  /* JSAMPLEs per row in output buffer */
  row_stride = width * num_components;


  /* Basic struct and information */
  if ((p_info = malloc(sizeof(frame_struct_t))) == NULL) {
    fprintf(stderr, "ERROR: Memory allocation failure\n");
    exit(1);
  }
  p_info->image_height = height;
  p_info->image_width = width;
  p_info->num_components = num_components;

  /* Image array and pointers to rows */
  if ((p_info->row_pointers = malloc(sizeof(JSAMPLE *) * height)) == NULL) {
    fprintf(stderr, "ERROR: Memory allocation failure\n");
    exit(1);
  }
  if ((p_info->image_buffer = malloc(sizeof(JSAMPLE) * row_stride * height)) == NULL) {
    fprintf(stderr, "ERROR: Memory allocation failure\n");
    exit(1);
  }
  for (i=0; i < height; i++)
    p_info->row_pointers[i] = & (p_info->image_buffer[i * row_stride]);

  printf("to\n");

  /* And send it back! */
  return p_info;
}
/*
 * write_JPEG_file writes out the contents of an image buffer to a JPEG.
 * A quality level of 2-100 can be provided (default = 75, high quality = ~95,
 * low quality = ~25, utter pixellation = 2).  Note that unlike read_JPEG_file,
 * it does not do any memory allocation on the buffer passed to it.
 */

void write_JPEG_file (char * filename, frame_ptr p_info, int quality){
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE * outfile;               /* target file */

  /* Step 1: allocate and initialize JPEG compression object */
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  /* Step 2: specify data destination (eg, a file) */
  /* Note: steps 2 and 3 can be done in either order. */

  if ((outfile = fopen(filename, "wb")) == NULL) {
    fprintf(stderr, "ERROR: Can't open output file %s\n", filename);
    exit(1);
  }
  jpeg_stdio_dest(&cinfo, outfile);

  /* Step 3: set parameters for compression */

  /* Set basic picture parameters (not optional) */
  cinfo.image_width = p_info->image_width;      /* image width and height, in pixels */

  cinfo.image_height = p_info->image_height;
  cinfo.input_components = p_info->num_components; /* # of color components per pixel */
  if (p_info->num_components == 3)
    cinfo.in_color_space = JCS_RGB;     /* colorspace of input image */
  else if (p_info->num_components == 1)
    cinfo.in_color_space = JCS_GRAYSCALE;
  else {
    fprintf(stderr, "ERROR: Non-standard colorspace for compressing!\n");
    exit(1);
  }
  /* Fill in the defaults for everything else, then override quality */
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);

  /* Step 4: Start compressor */
  jpeg_start_compress(&cinfo, TRUE);

  /* Step 5: while (scan lines remain to be written) */
  /*           jpeg_write_scanlines(...); */
  while (cinfo.next_scanline < cinfo.image_height) {
    (void) jpeg_write_scanlines(&cinfo, &(p_info->row_pointers[cinfo.next_scanline]), 1);
  }
  /* Step 6: Finish compression & close output */

  jpeg_finish_compress(&cinfo);
  fclose(outfile);

  /* Step 7: release JPEG compression object */
  jpeg_destroy_compress(&cinfo);
}

void destroy_frame(frame_ptr kill_me){
  free(kill_me->image_buffer);
  free(kill_me->row_pointers);
  free(kill_me);
}

int gammaEncode(float radiance, float d){
  float max = (radiance * d) > 0 ? radiance * d : 0.0f;
  float min = max < 1 ? max : 1.0f;
  int temp = (int)(pow(min, 1.0f/2.2f) * 255.0f);
  return temp;
}

int main(int argc, char **argv){

  float d = 15.0f;

  if(argc != 4){
    fprintf(stderr, "Usage: ray-tracer [input.txt] [output.jpg]\n");
    exit(1);
  }

  printf("main method hello\n");

  frame_ptr to = allocate_frame(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_COMPONENTS);

  printf("hello to\n");

  // Image to be rasterized
  image_ptr image = render(argv);

  // translate raster Image to frame by iterating through pixels
  for(int i = 0; i < image->height; i++){

    for(int j = 0; j < image->width; j++){

      // set RGB values
      to->row_pointers[i][j * NUM_COMPONENTS] = gammaEncode(image->data[(i * image->width) + j][0], d);
      to->row_pointers[i][j * NUM_COMPONENTS + 1] =  gammaEncode(image->data[(i * image->width) + j][1], d);
      to->row_pointers[i][j * NUM_COMPONENTS + 2] =  gammaEncode(image->data[(i * image->width) + j][2], d);

    }
  }

  write_JPEG_file(argv[2], to, OUT_QUALITY);
  destroy_frame(to);

  for(int i = 0; i < image->width * image->height; i++){
    free(image->data[i]);
  }

  free(image->data);
  free(image);

}
