//
//  filter.c
//
//
//  Created by Julia Tucher on 11/26/19.
//

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(int argc, char **argv){

  if(argc != 3){
    fprintf(stderr, "Usage: filter [input.txt] [output.txt]\n");
    exit(1);
  }

  FILE *stream = fopen(argv[1], "r");
  FILE *output = fopen(argv[2], "w");

  int the;
  char temp;
  int tempd;
  int size = 100;
  int n_count = 0;
  int v_count = 0;
  float *vertices = (float *)malloc(size * sizeof(float));
  float *normals = (float *)malloc(size * sizeof(float));

  int index_v[3];
  int index_n[3];

  fscanf(stream, "%c", &temp);
  while(temp != 10){
    printf("output %c\n", temp);
    assert(temp != 10);
    if(temp == 'v'){
      if(3*v_count + 2 >= size){
	  size *= 2;
	  vertices = (float *)realloc(vertices, sizeof(float) * size);
	  normals = (float *)realloc(normals, sizeof(float) * size);
	}
	
      for(int i = 0; i < 3; i++){
	fscanf(stream, "%f", &vertices[3*v_count + i]);
      }
      v_count++;
      fscanf(stream, "%c", &temp);
    }
    else if(temp == 'n'){
	if(3*n_count + 3 >= size){
	  size *= 2;
	  vertices = (float *)realloc(vertices, sizeof(float) * size);
	  normals = (float *)realloc(normals, sizeof(float) * size);
	}
      for(int i = 0; i < 3; i++){
	fscanf(stream, "%f", &normals[3*n_count + i]);
      }
      n_count++;
      fscanf(stream, "%c", &temp);
    }
    else if(temp == 'f'){
      for(int i = 0; i < 3; i++){
	fscanf(stream, "%d", &index_v[i]);
	fscanf(stream, "%c", &temp);
	fscanf(stream, "%d", &tempd);
	fscanf(stream, "%c", &temp);
	fscanf(stream, "%d", &index_n[i]);
	
      }

      for(int i = 0; i < 3; i++){
	printf("here %d\n", i);
	if(index_v[i] < 0){
	  index_v[i] += v_count;
	}
	for(int j = 0; j < 3; j++){
	  fprintf(output, "%f ", vertices[3*index_v[i] + j]);
	}
      }
      fputc('\n', output);
      for(int i = 0; i < 3; i++){
	if(index_n[i] < 0){
	  index_n[i] += n_count;
	}
	
	for(int j = 0; j < 3; j++){
	  float tempf =  normals[3*index_n[i] + j];
	fprintf(output, "%f ", tempf);
	}
      }
      fputc('\n', output);
      fscanf(stream, "%c", &temp);
      fprintf(output, "1 2.5 2.0 0.0 0.2 0.2 0.2 100\n\n");
    }
    else{
      do{
	fscanf(stream, "%c", &temp);
      }while(temp != '\n');
    }
    fscanf(stream, "%c", &temp);
  }

  fprintf(output, "1.0 3.0 1.0 10.0 10.0 10.0\n");
      
  fclose(output);
  fclose(stream);

}
