// TODO: split this into hpp and cpp, but the first attempt failed
//#include "../mfem.hpp"
//extern class mfem::Vector;
using namespace mfem;
//#include "../linalg/vector.hpp"

double uFunTest_ex(const Vector& x); // Exact Solution
double uFunTest_ex_dt(const Vector& xt);
double uFunTest_ex_dt2(const Vector & xt);
double uFunTest_ex_laplace(const Vector & xt);
double uFunTest_ex_dtlaplace(const Vector & xt);
void uFunTest_ex_gradx(const Vector& xt, Vector& grad);
void uFunTest_ex_dtgradx(const Vector& xt, Vector& gradx );

void bFunRect2D_ex(const Vector& xt, Vector& b );
double  bFunRect2Ddiv_ex(const Vector& xt);

void bFunCube3D_ex(const Vector& xt, Vector& b );
double  bFunCube3Ddiv_ex(const Vector& xt);

void bFunSphere3D_ex(const Vector& xt, Vector& b );
double  bFunSphere3Ddiv_ex(const Vector& xt);

void bFunCircle2D_ex (const Vector& xt, Vector& b);
double  bFunCircle2Ddiv_ex(const Vector& xt);

double uFunTest_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double res = t*t*exp(t) * sin (3.0 * M_PI * x) * sin (2.0 * M_PI * y);
    if (xt.Size() == 4)
        res *= sin (M_PI * z);

    return res;
}

double uFunTest_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double res = (t*t + 2.0 * t)*exp(t) * sin (3.0 * M_PI * x) * sin (2.0 * M_PI * y);
    if (xt.Size() == 4)
        res *= sin (M_PI * z);

    return res;
}

double uFunTest_ex_dt2(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double res = ((t*t + 2.0 * t) + (2.0 + 2.0 * t))*exp(t) * sin (3.0 * M_PI * x) * sin (2.0 * M_PI * y);
    if (xt.Size() == 4)
        res *= sin (M_PI * z);

    return res;
}

double uFunTest_ex_laplace(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double res = sin (3.0 * M_PI * x) * sin (2.0 * M_PI * y) * M_PI * M_PI;
    if (xt.Size() == 3)
        res *= (-1) * (2.0 * 2.0 + 3.0 * 3.0);
    else
    {
        res *= sin (M_PI * z);
        res *= (-1) * (2.0 * 2.0 + 3.0 * 3.0 + 1.0  * 1.0);
    }
    res *= t*t*exp(t);

    return res;
}

double uFunTest_ex_dtlaplace(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double res = sin (3.0 * M_PI * x) * sin (2.0 * M_PI * y) * M_PI * M_PI;
    if (xt.Size() == 3)
        res *= (-1) * (2.0 * 2.0 + 3.0 * 3.0);
    else
    {
        res *= sin (M_PI * z);
        res *= (-1) * (2.0 * 2.0 + 3.0 * 3.0 + 1.0  * 1.0);
    }
    res *= (t*t + 2.0 * t)*exp(t);

    return res;
}

void uFunTest_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = t*t*exp(t) * 3.0 * M_PI * cos (3.0 * M_PI * x) * sin (2.0 * M_PI * y);
    gradx(1) = t*t*exp(t) * sin (3.0 * M_PI * x) * 2.0 * M_PI * cos ( 2.0 * M_PI * y);
    if (xt.Size() == 4)
    {
        gradx(0) *= sin (M_PI * z);
        gradx(1) *= sin (M_PI * z);
        gradx(2) = t*t*exp(t) * sin (3.0 * M_PI * x) * sin ( 2.0 * M_PI * y) * M_PI * cos (M_PI * z);
    }
}

void uFunTest_ex_dtgradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = (t*t + 2.0 * t)*exp(t) * 3.0 * M_PI * cos (3.0 * M_PI * x) * sin (2.0 * M_PI * y);
    gradx(1) = (t*t + 2.0 * t)*exp(t) * sin (3.0 * M_PI * x) * 2.0 * M_PI * cos ( 2.0 * M_PI * y);
    if (xt.Size() == 4)
    {
        gradx(0) *= sin (M_PI * z);
        gradx(1) *= sin (M_PI * z);
        gradx(2) = (t*t + 2.0 * t)*exp(t) * sin (3.0 * M_PI * x) * sin ( 2.0 * M_PI * y) * M_PI * cos (M_PI * z);
    }
}

// velocity for hyperbolic problems
void bFunRect2D_ex(const Vector& xt, Vector& b )
{
    double x = xt(0);
    double y = xt(1);

    b.SetSize(xt.Size());

    b(0) = sin(x*M_PI)*cos(y*M_PI);
    b(1) = - sin(y*M_PI)*cos(x*M_PI);

    b(xt.Size()-1) = 1.;

}

double bFunRect2Ddiv_ex(const Vector& xt)
{
    return 0.0;
}

void bFunCube3D_ex(const Vector& xt, Vector& b )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);

    b.SetSize(xt.Size());

    b(0) = sin(x*M_PI)*cos(y*M_PI)*cos(z*M_PI);
    b(1) = - 0.5 * sin(y*M_PI)*cos(x*M_PI) * cos(z*M_PI);
    b(2) = - 0.5 * sin(z*M_PI)*cos(x*M_PI) * cos(y*M_PI);

    b(xt.Size()-1) = 1.;

}

double bFunCube3Ddiv_ex(const Vector& xt)
{
    return 0.0;
}

void bFunSphere3D_ex(const Vector& xt, Vector& b )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    b.SetSize(xt.Size());

    b(0) = -y;  // -x2
    b(1) = x;   // x1
    b(2) = 0.0;

    b(xt.Size()-1) = 1.;
    return;
}

double bFunSphere3Ddiv_ex(const Vector& xt)
{
    return 0.0;
}
void bFunCircle2D_ex(const Vector& xt, Vector& b )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    b.SetSize(xt.Size());

    b(0) = -y;  // -x2
    b(1) = x;   // x1

    b(xt.Size()-1) = 1.;
    return;
}

double bFunCircle2Ddiv_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return 0.0;
}

