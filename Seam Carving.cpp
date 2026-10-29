#include<iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


Mat calculate_energy(const Mat &img)
{
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat grad_x, grad_y;
    Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    Sobel(gray, grad_y, CV_32F, 0, 1, 3);

    Mat energy;
    magnitude(grad_x, grad_y, energy);

    return energy;
}

void draw_vertical_seam(Mat &pic, int *seam)
{
    for (int r = 0; r < pic.rows; r++) 
    {
        int c = seam[r];   
        if (c >= 0 && c < pic.cols) 
        {
            pic.at<Vec3b>(r,c) = Vec3b(0,0,255);
        }
    }
}

void draw_horizontal_seam(Mat &pic, int *seam)
{
    for (int c = 0; c < pic.cols; c++) 
    {
        int r = seam[c];   
        if (r >= 0 && r < pic.rows) 
        {
            pic.at<Vec3b>(r,c) = Vec3b(0,0,255);
        }
    }
}

void find_vertical_seam(const Mat &energy, int *seam)
{
    int height = energy.rows;
    int width = energy.cols;

    float **min_energy = new float*[height];
    int **source = new int*[height];

    for (int i = 0; i < height; i++) 
    {
        min_energy[i] = new float[width];
        source[i] = new int[width];
    }
    for(int i=0;i<width;i++)
    {
        min_energy[0][i] = energy.at<float>(0, i);
        source[0][i] = -1;
    }


    for(int j=1;j<height;j++)
    {
        for(int i=0;i<width;i++)
        {
            float min_prev = min_energy[j-1][i];
            int temp = i;

            if(i>0 && min_energy[j-1][i-1] < min_prev)
            {
                min_prev = min_energy[j-1][i-1];
                temp = i-1;
            }
            if(i<width-1 && min_energy[j-1][i+1] < min_prev)
            {
                min_prev = min_energy[j-1][i+1];
                temp = i+1;
            }
            min_energy[j][i] = energy.at<float>(j, i) + min_prev;
            source[j][i] = temp;
        }
    }

    int min_index = 0;
    float min_value = min_energy[height-1][0];
    for(int i=1;i<width;i++)
    {
        if(min_energy[height-1][i] < min_value)
        {
            min_index = i;
            min_value = min_energy[height-1][i];
        }
    }

    seam[height-1] = min_index;
    for(int i=height-1; i>0; i--)
    {
        seam[i-1] = source[i][seam[i]];
    }
    
    for (int i = 0; i < height; i++) {
        delete[] min_energy[i];
        delete[] source[i];
    }
    delete[] min_energy;
    delete[] source;
};

Mat vertical_seam_removal(const Mat &image, int *seam)
{
    int height = image.rows;
    int width = image.cols;

    Mat updated_image(height, width-1, image.type());

    for(int j=0;j<height;j++)
    {
        int pos=0;
        for(int i=0;i<width;i++)
        {
            if(i==seam[j])
                continue;
            updated_image.at<Vec3b>(j,pos++) = image.at<Vec3b>(j,i);
        }
    }
    return updated_image;
}

void find_horizontal_seam(const Mat &energy, int *seam)
{
    int height = energy.rows;
    int width = energy.cols;

    float **min_energy = new float*[height];
    int **source = new int*[height];
    for (int i = 0; i < height; i++) 
    {
        min_energy[i] = new float[width];
        source[i] = new int[width];
    }
    for(int j=0;j<height;j++)
    {
        min_energy[j][0] = energy.at<float>(j,0);
        source[j][0] = -1;
    }


    for(int i=1;i<width;i++)
    {
        for(int j=0;j<height;j++)
        {
            float min_prev = min_energy[j][i-1];
            int min_index = j;

            if(j>0 && min_energy[j-1][i-1]<min_prev)
            {
                min_prev = min_energy[j-1][i-1];
                min_index = j-1;
            }
            if(j<height-1 && min_energy[j+1][i-1] < min_prev)
            {
                min_prev = min_energy[j+1][i-1];
                min_index = j+1;
            }
            min_energy[j][i] = energy.at<float>(j,i) + min_prev;
            source[j][i] = min_index;
        }
    }

    int min_index = 0;
    float min_value = min_energy[0][width-1];
    for(int i=1;i<height;i++)
    {
        if(min_energy[i][width-1] < min_value)
        {
            min_value = min_energy[i][width-1];
            min_index = i;
        }
    }

    seam[width-1] = min_index;
    for(int i=width-1; i>0; i--)
    {
        seam[i-1] = source[ seam[i] ][ i ];
    }

    for (int i = 0; i < height; i++) {
        delete[] min_energy[i];
        delete[] source[i];
    }
    delete[] min_energy;
    delete[] source;
}

Mat horizontal_seam_removal(const Mat &image, int *seam)
{
    int height = image.rows;
    int width = image.cols;

    Mat updated_image(height-1, width, image.type());

    for(int j=0;j<width;j++)
    {
        int pos=0;
        for(int i=0;i<height;i++)
        {
            if(i==seam[j])
                continue;
            updated_image.at<Vec3b>(pos++,j) = image.at<Vec3b>(i,j);
        }
    }
    return updated_image;
}

int main()
{
    string pic_name;
    cout<<"Enter the name of picture:"<<endl;
    cin>>pic_name;

    Mat image = imread(pic_name, IMREAD_COLOR);
    if(image.empty())
    {
        cout<<"Picture "<< pic_name<<" could not be open or found.";
        return -1;
    }

    int final_width, final_height;
    cout<<"Required size of the image:"<<endl;
    cin>>final_width;
    cin>>final_height;

    while(image.cols>final_width || image.rows > final_height)
    {
        int height = image.rows;
        int width  = image.cols;

        if(width > final_width)
        {
            int *seam = new int[height];
            Mat energy = calculate_energy(image);
            find_vertical_seam(energy, seam);
            Mat temp = image.clone();
            draw_vertical_seam(temp, seam);
            imshow("Seam Carving", temp);
            waitKey(100);
            image = vertical_seam_removal(image, seam);
            delete[] seam;
        }
        else if(height > final_height)
        {
            int *seam = new int[width];
            Mat energy = calculate_energy(image);
            find_horizontal_seam(energy, seam);
            Mat temp = image.clone();
            draw_horizontal_seam(temp, seam);
            imshow("Seam Carving",temp);
            waitKey(100);
            image = horizontal_seam_removal(image, seam);
            delete[] seam;
        }
    }

    imshow("Seam Carving",image);
    waitKey(0);
    return 0;
}
