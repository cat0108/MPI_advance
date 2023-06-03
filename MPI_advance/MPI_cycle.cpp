#include <iostream>
#include <mpi.h>
#include<Windows.h>
#include<omp.h>
#include<cstring>
using namespace std;

float matrix[10000][10000];
float matrix1[10000][10000];
float matrix2[10000][10000];
float matrix3[10000][10000];
int Num_thread = 4;//ÿ������4�̱߳��
void Initialize(int N)
{
    for (int i = 0; i < N; i++)
    {
        //���Ƚ�ȫ��Ԫ����Ϊ0���Խ���Ԫ����Ϊ1
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0;
            matrix1[i][j] = 0;
            matrix2[i][j] = 0;
            matrix3[i][j] = 0;
        }
        matrix[i][i] = matrix1[i][i] = matrix2[i][i] = matrix3[i][i] = 1.0;
        //�������ǵ�λ�ó�ʼ��Ϊ�����
        for (int j = i + 1; j < N; j++)
        {
            matrix[i][j] = matrix1[i][j] = matrix2[i][j] = matrix3[i][j] = rand();
        }
    }
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                matrix[i][j] += matrix[k][j];
                matrix1[i][j] += matrix1[k][j];
                matrix2[i][j] += matrix2[k][j];
                matrix3[i][j] += matrix3[k][j];

            }
        }
    }
}
//ƽ���㷨
void normal(float matrix[][10000], int N) {
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++)
            matrix[k][j] /= matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++)
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}
//һάѭ������
void MPI_cycle(float matrix[][10000], int N) {
    int rank;
    int size;
    double begin_time, end_time;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
        begin_time = MPI_Wtime();
    else
        begin_time = 0;

    // ����Ԫ����
    for (int k = 0; k < N; k++)
    {
        // �鿴�Ƿ��ɱ����̸����������
        if (k%size==rank)
        {
            for (int j = k + 1; j < N; j++)
                matrix[k][j] /= matrix[k][k];
            matrix[k][k] = 1;
            for (int p = 0; p < size; p++) {
                if (p != rank)
                    MPI_Send(&matrix[k][0], N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);//�㲥����������
            }
        }
        // ������̽���
        else {
            MPI_Recv(&matrix[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // ������Ԫ����,��ĳ�еķ�Χ�ڲŸ���
        for (int i = k+k%size; i < N; i+=size)
        {
            for (int j = k + 1; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //��ȥ��ɣ����лش�����,����ش���0�Ž���

    if (rank != 0)
        for (int i = rank; i < N; i += size)
            MPI_Send(&matrix[i][0], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

    else//��������
        for (int i = 1; i < size; i++)
            for (int j = i; j < N; j+=size)
                MPI_Recv(&matrix[j][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    end_time = MPI_Wtime();
    if (rank == 0)
        cout << "MPI_cycle time: " << (end_time - begin_time) * 1000 << " ms" << endl;

}

//��ˮ�߻���
void MPI_pipeLine(float matrix[][10000], int N) {
    int rank;
    int size;
    double begin_time, end_time;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
        begin_time = MPI_Wtime();
    else
        begin_time = 0;
    int part = N / size;//ÿ�����̷��������
    int start = rank * part;
    int end = start + part - 1;
    if (rank == size - 1) {
        end = N - 1;
        part = N - start;//���һ�����̷��������
    }
    // ����Ԫ����
    for (int k = 0; k < N; k++)
    {
        // �鿴�Ƿ��ɱ����̸����������
        if (k >= start && k <= end)
        {
            for (int j = k + 1; j < N; j++)
                matrix[k][j] /= matrix[k][k];      
            if (rank < size - 1)//����������һ�����̣��͹㲥����һ������
                MPI_Send(&matrix[k][0], N, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
        }
        // ֮��Ľ��̽�������ǰһ�����̵���Ϣ,��������������͸���һ������
        else {      
            if (k < end)
            {
                MPI_Recv(&matrix[k][0], N, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (rank < size - 1)
                    MPI_Send(&matrix[k][0], N, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
            }
        }
        // ������Ԫ����,��ĳ�еķ�Χ�ڲŸ���
        for (int i = max(k + 1, start); i <= end; i++)
        {
            for (int j = k + 1; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //��ȥ��ɣ����лش�����,����ش���0�Ž���
    if (rank != 0)
        MPI_Send(&matrix[start][0], part * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    else//���������̽��ս�������
        for (int i = 1; i < size; i++)
            MPI_Recv(&matrix[end + 1 + (i - 1) * part][0], part * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    end_time = MPI_Wtime();
    if (rank == 0)
        cout << "MPI_pipeline time: " << (end_time - begin_time) * 1000 << " ms" << endl;

}

//ѭ�����ֺ�openmp���
void MPI_omp(float matrix[][10000], int N) {
    int rank;
    int size;
    double begin_time, end_time;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
        begin_time = MPI_Wtime();
    else
        begin_time = 0;
    int part = N / size;//ÿ�����̷��������
    int start = rank * part;
    int end = start + part - 1;
    if (rank == size - 1) {
        end = N - 1;
        part = N - start;//���һ�����̷��������
    }
    // ����Ԫ����
#pragma omp parallel num_threads(Num_thread) shared(matrix,rank,size,N)
    for (int k = 0; k < N; k++)
    {
        // �鿴�Ƿ��ɱ����̸����������
#pragma omp single
        {
            if (k >= start && k <= end)
            {
                for (int j = k + 1; j < N; j++)
                    matrix[k][j] /= matrix[k][k];
                matrix[k][k] = 1;
                for (int p = 0; p < size; p++) {
                    if (p != rank)
                        MPI_Send(&matrix[k][0], N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);//�㲥����������
                }
            }
            // ������̽���
            else {
                MPI_Recv(&matrix[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        // ������Ԫ����,��ĳ�еķ�Χ�ڲŸ���
#pragma omp for schedule(guided)
            for (int i = max(k + 1, start); i <= end; i++)
            {
                for (int j = k + 1; j < N; j++) {
                    matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
                }
                matrix[i][k] = 0;
            }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //��ȥ��ɣ����лش�����,����ش���0�Ž���
    if (rank != 0)
        MPI_Send(&matrix[start][0], part * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    else//���������̽��ս�������
        for (int i = 1; i < size; i++)
            MPI_Recv(&matrix[end + 1 + (i - 1) * part][0], part * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    end_time = MPI_Wtime();
    if (rank == 0)
        cout << "MPI_omp time: " << (end_time - begin_time) * 1000 << " ms" << endl;

}

int main() {
    int N = 3000;
    LARGE_INTEGER fre, begin, end;
    double gettime;
    int rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    QueryPerformanceFrequency(&fre);
    QueryPerformanceCounter(&begin);
    Initialize(N);
    QueryPerformanceCounter(&end);
    gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        cout << "intial time: " << gettime << " ms" << endl;


    QueryPerformanceFrequency(&fre);
    QueryPerformanceCounter(&begin);
    normal(matrix, N);
    QueryPerformanceCounter(&end);
    gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        cout << "normal time: " << gettime << " ms" << endl;


    MPI_cycle(matrix1, N);
    MPI_pipeLine(matrix2, N);
    MPI_omp(matrix3, N);
    MPI_Finalize();

}