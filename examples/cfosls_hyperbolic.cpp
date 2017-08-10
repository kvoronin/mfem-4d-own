//                                MFEM(with 4D elements) CFOSLS with S from H1 for 3D/4D hyperbolic equation
//                                  with mesh generator and visualization
//
// Compile with: make
//
// Description:  This example code solves a simple 3D/4D hyperbolic problem over [0,1]^3(4)
//               corresponding to the saddle point system
//                                  sigma_1 = u * b
//							 		sigma_2 - u        = 0
//                                  div_(x,t) sigma    = f
//                       with b = vector function (~velocity),
//						 NO boundary conditions (which work only in case when b * n = 0 pointwise at the domain space boundary)
//						 and initial condition:
//                                  u(x,0)            = 0
//               Here, we use a given exact solution
//                                  u(xt) = uFun_ex(xt)
//               and compute the corresponding r.h.s.
//               We discretize with Raviart-Thomas finite elements (sigma), continuous H1 elements (u) and
//					  discontinuous polynomials (mu) for the lagrange multiplier.
//
// Solver: MINRES preconditioned by boomerAMG or ADS

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

#include"cfosls_testsuite.hpp"

#define MYZEROTOL (1.0e-13)

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

//********* NEW STUFF FOR 4D CFOSLS
//-----------------------
/// Integrator for (Q u, v) for VectorFiniteElements

class PAUVectorFEMassIntegrator: public BilinearFormIntegrator
{
private:
   Coefficient *Q;
   VectorCoefficient *VQ;
   MatrixCoefficient *MQ;
   void Init(Coefficient *q, VectorCoefficient *vq, MatrixCoefficient *mq)
   { Q = q; VQ = vq; MQ = mq; }

#ifndef MFEM_THREAD_SAFE
   Vector shape;
   Vector D;
   Vector test_shape;
   Vector b;
   DenseMatrix trial_vshape;
#endif

public:
   PAUVectorFEMassIntegrator() { Init(NULL, NULL, NULL); }
   PAUVectorFEMassIntegrator(Coefficient *_q) { Init(_q, NULL, NULL); }
   PAUVectorFEMassIntegrator(Coefficient &q) { Init(&q, NULL, NULL); }
   PAUVectorFEMassIntegrator(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
   PAUVectorFEMassIntegrator(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
   PAUVectorFEMassIntegrator(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
   PAUVectorFEMassIntegrator(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

//=-=-=-=--=-=-=-=-=-=-=-=-=
/// Integrator for (Q u, v) for VectorFiniteElements
class PAUVectorFEMassIntegrator2: public BilinearFormIntegrator
{
private:
   Coefficient *Q;
   VectorCoefficient *VQ;
   MatrixCoefficient *MQ;
   void Init(Coefficient *q, VectorCoefficient *vq, MatrixCoefficient *mq)
   { Q = q; VQ = vq; MQ = mq; }

#ifndef MFEM_THREAD_SAFE
   Vector shape;
   Vector D;
   Vector trial_shape;
   Vector test_shape;//<<<<<<<
   DenseMatrix K;
   DenseMatrix test_vshape;
   DenseMatrix trial_vshape;
   DenseMatrix trial_dshape;//<<<<<<<<<<<<<<
   DenseMatrix test_dshape;//<<<<<<<<<<<<<<
   DenseMatrix dshape;
   DenseMatrix dshapedxt;
   DenseMatrix invdfdx;
#endif

public:
   PAUVectorFEMassIntegrator2() { Init(NULL, NULL, NULL); }
   PAUVectorFEMassIntegrator2(Coefficient *_q) { Init(_q, NULL, NULL); }
   PAUVectorFEMassIntegrator2(Coefficient &q) { Init(&q, NULL, NULL); }
   PAUVectorFEMassIntegrator2(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
   PAUVectorFEMassIntegrator2(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
   PAUVectorFEMassIntegrator2(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
   PAUVectorFEMassIntegrator2(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

//=-=-=-=-=-=-=-=-=-=-=-=-=-



void PAUVectorFEMassIntegrator::AssembleElementMatrix(
        const FiniteElement &el,
        ElementTransformation &Trans,
        DenseMatrix &elmat)
{}


void PAUVectorFEMassIntegrator::AssembleElementMatrix2(
        const FiniteElement &trial_fe, const FiniteElement &test_fe,
        ElementTransformation &Trans, DenseMatrix &elmat)
{
    // assume both test_fe is vector FE, trial_fe is scalar FE
    int dim  = test_fe.GetDim();
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();
    double w;

    if (VQ == NULL) // || = or
        mfem_error("VectorFEMassIntegrator::AssembleElementMatrix2(...)\n"
                "   is not implemented for non-vector coefficients");

#ifdef MFEM_THREAD_SAFE
    Vector trial_shape(trial_dof);
    DenseMatrix test_vshape(test_dof,dim);
#else
    trial_vshape.SetSize(trial_dof,dim);
    test_shape.SetSize(test_dof);
#endif
    elmat.SetSize (test_dof, trial_dof);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        int order = (Trans.OrderW() + test_fe.GetOrder() + trial_fe.GetOrder());
        ir = &IntRules.Get(test_fe.GetGeomType(), order);
    }

    elmat = 0.0;
//    b.SetSize(dim);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        test_fe.CalcShape(ip, test_shape);

        Trans.SetIntPoint (&ip);
        trial_fe.CalcVShape(Trans, trial_vshape);

        w = ip.weight * Trans.Weight();
        VQ->Eval (b, Trans, ip);

        for (int j = 0; j < trial_dof; j++)
            for (int k = 0; k < test_dof; k++)
                for (int d = 0; d < dim; d++ )
                    elmat(k, j) += w*trial_vshape(j,d)*b(d)*test_shape(k);
    }
}
///////////////////////////

void PAUVectorFEMassIntegrator2::AssembleElementMatrix(
        const FiniteElement &el,
        ElementTransformation &Trans,
        DenseMatrix &elmat)
{
    int dof = el.GetDof();
    int dim  = el.GetDim();
    double w;

    if (VQ || MQ) // || = or
        mfem_error("VectorFEMassIntegrator::AssembleElementMatrix2(...)\n"
                "   is not implemented for vector/tensor permeability");

#ifdef MFEM_THREAD_SAFE
    Vector shape(dof);
    DenseMatrix dshape(dof,dim);
    DenseMatrix dshapedxt(dof,dim);
    DenseMatrix invdfdx(dim,dim);
#else
    shape.SetSize(dof);
    dshape.SetSize(dof,dim);
    dshapedxt.SetSize(dof,dim);
    invdfdx.SetSize(dim,dim);
#endif
    //elmat.SetSize (test_dof, trial_dof);
    elmat.SetSize (dof, dof);
    elmat = 0.0;

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        int order = (Trans.OrderW() + el.GetOrder() + el.GetOrder());
        ir = &IntRules.Get(el.GetGeomType(), order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        //chak Trans.SetIntPoint (&ip);

        el.CalcShape(ip, shape);
        el.CalcDShape(ip, dshape);

        Trans.SetIntPoint (&ip);
        CalcInverse(Trans.Jacobian(), invdfdx);
        w = ip.weight * Trans.Weight();
        Mult(dshape, invdfdx, dshapedxt);

        if (Q)
        {
            w *= Q -> Eval (Trans, ip);
        }

        for (int j = 0; j < dof; j++)
        {
            for (int k = 0; k < dof; k++)
            {
                for (int d = 0; d < dim - 1; d++ )
                    elmat(j, k) +=  w * dshapedxt(j, d) * dshapedxt(k, d);
                elmat(j, k) +=  w * shape(j) * shape(k);
            }
        }
    }
}

void PAUVectorFEMassIntegrator2::AssembleElementMatrix2(
        const FiniteElement &trial_fe, const FiniteElement &test_fe,
        ElementTransformation &Trans, DenseMatrix &elmat)
{}

//********* END OF NEW STUFF FOR CFOSLS 4D

//********* NEW STUFF FOR 4D CFOSLS
//---------
class VectordivDomainLFIntegrator : public LinearFormIntegrator
{
   Vector divshape;
   Coefficient &Q;
   int oa, ob;
public:
   /// Constructs a domain integrator with a given Coefficient
   VectordivDomainLFIntegrator(Coefficient &QF, int a = 2, int b = 0)
   // the old default was a = 1, b = 1
   // for simple elliptic problems a = 2, b = -2 is ok
      : Q(QF), oa(a), ob(b) { }

   /// Constructs a domain integrator with a given Coefficient
   VectordivDomainLFIntegrator(Coefficient &QF, const IntegrationRule *ir)
      : LinearFormIntegrator(ir), Q(QF), oa(1), ob(1) { }

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};
//---------

//------------------
void VectordivDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)//don't need the matrix but the vector
{
   int dof = el.GetDof();

   divshape.SetSize(dof);       // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob + Tr.OrderW());
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
     // int order = 2 * el.GetOrder() ; // <--- OK for RTk
     // ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDivShape(ip, divshape);

      Tr.SetIntPoint (&ip);
//      double val = Tr.Weight() * Q.Eval(Tr, ip);
      // Chak: Looking at how MFEM assembles in VectorFEDivergenceIntegrator, I think you dont need Tr.Weight() here
      // I think this is because the RT (or other vector FE) basis is scaled by the geometry of the mesh
      double val = Q.Eval(Tr, ip);

      add(elvect, ip.weight * val, divshape, elvect);
   }
}

class GradDomainLFIntegrator : public LinearFormIntegrator
{
   DenseMatrix dshape;
   DenseMatrix invdfdx;
   DenseMatrix dshapedxt;
   Vector bf;
   Vector bfdshapedxt;
   VectorCoefficient &Q;
   int oa, ob;
public:
   /// Constructs a domain integrator with a given Coefficient
   GradDomainLFIntegrator(VectorCoefficient &QF, int a = 2, int b = 0)
   // the old default was a = 1, b = 1
   // for simple elliptic problems a = 2, b = -2 is ok
      : Q(QF), oa(a), ob(b) { }

   /// Constructs a domain integrator with a given Coefficient
   GradDomainLFIntegrator(VectorCoefficient &QF, const IntegrationRule *ir)
      : LinearFormIntegrator(ir), Q(QF), oa(1), ob(1) { }

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

void GradDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   int dim  = el.GetDim();

   dshape.SetSize(dof,dim);       // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   invdfdx.SetSize(dim,dim);
   dshapedxt.SetSize(dof,dim);
   bf.SetSize(dim);
   bfdshapedxt.SetSize(dof);
   double w;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
//       ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob
//                          + Tr.OrderW());
//      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
     // int order = 2 * el.GetOrder() ; // <--- OK for RTk
      int order = (Tr.OrderW() + el.GetOrder() + el.GetOrder());
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

//      Tr.SetIntPoint (&ip);
      //double val = Tr.Weight() * Q.Eval(Tr, ip);

      Tr.SetIntPoint (&ip);
      w = ip.weight;// * Tr.Weight();
      CalcAdjugate(Tr.Jacobian(), invdfdx);
      Mult(dshape, invdfdx, dshapedxt);

      Q.Eval(bf, Tr, ip);

      dshapedxt.Mult(bf, bfdshapedxt);

      /*
      cout << "Jacobian" << endl;
      Tr.Jacobian().Print();

      cout << "invdfdx" << endl;
      invdfdx.Print();

      cout << "dshape" << endl;
      dshape.Print();

      cout << "dshapedxt" << endl;
      dshapedxt.Print();

      cout << "bf" << endl;
      bf.Print();
      */

      add(elvect, w, bfdshapedxt, elvect);
      //cout << "elvect = " << elvect << endl;
   }
}

//------------------
//********* END OF NEW STUFF FOR CFOSLS 4D


//------------------
//********* END OF NEW BilinearForm and LinearForm integrators FOR CFOSLS 4D (used only for heat equation, so can be deleted)

namespace mfem
{

/// class for function coefficient with parameters
class FunctionCoefficientExtra : public Coefficient
{
private:
    double * parameters;
    int nparams;

protected:
   double (*Function)(const Vector &, double *, const int&);

public:
   /// Define a time-independent coefficient from a C-function
   FunctionCoefficientExtra(double (*f)(const Vector &, double *, const int&), double * Parameters, int Nparams)
   {
      Function = f;
      nparams = Nparams;
      parameters = new double[nparams];
      for ( int i = 0; i < nparams; ++i)
          parameters[i] = Parameters[i];
   }

   /// Evaluate coefficient
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

double FunctionCoefficientExtra::Eval(ElementTransformation & T,
                                 const IntegrationPoint & ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   if (Function)
   {
      return ((*Function)(transip, parameters, nparams));
   }
}
}

double uFun_ex(const Vector& x); // Exact Solution
double uFun_ex_dt(const Vector& xt);
void uFun_ex_gradx(const Vector& xt, Vector& grad);

void bFun_ex (const Vector& xt, Vector& b);
double  bFundiv_ex(const Vector& xt);

double uFun3_ex(const Vector& x); // Exact Solution
double uFun3_ex_dt(const Vector& xt);
void uFun3_ex_gradx(const Vector& xt, Vector& grad);

double uFun4_ex(const Vector& x); // Exact Solution
double uFun4_ex_dt(const Vector& xt);
void uFun4_ex_gradx(const Vector& xt, Vector& grad);

//void bFun4_ex (const Vector& xt, Vector& b);

//void bFun6_ex (const Vector& xt, Vector& b);

double uFun5_ex(const Vector& x); // Exact Solution
double uFun5_ex_dt(const Vector& xt);
void uFun5_ex_gradx(const Vector& xt, Vector& grad);

double uFun6_ex(const Vector& x); // Exact Solution
double uFun6_ex_dt(const Vector& xt);
void uFun6_ex_gradx(const Vector& xt, Vector& grad);

double uFunCylinder_ex(const Vector& x); // Exact Solution
double uFunCylinder_ex_dt(const Vector& xt);
void uFunCylinder_ex_gradx(const Vector& xt, Vector& grad);

double uFun66_ex(const Vector& x); // Exact Solution
double uFun66_ex_dt(const Vector& xt);
void uFun66_ex_gradx(const Vector& xt, Vector& grad);


double uFun2_ex(const Vector& x); // Exact Solution
double uFun2_ex_dt(const Vector& xt);
void uFun2_ex_gradx(const Vector& xt, Vector& grad);

void Hdivtest_fun(const Vector& xt, Vector& out );
double  L2test_fun(const Vector& xt);

double uFun33_ex(const Vector& x); // Exact Solution
double uFun33_ex_dt(const Vector& xt);
void uFun33_ex_gradx(const Vector& xt, Vector& grad);

double uFun10_ex(const Vector& x); // Exact Solution
double uFun10_ex_dt(const Vector& xt);
void uFun10_ex_gradx(const Vector& xt, Vector& grad);

template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaTemplate(const Vector& xt, Vector& sigma);
template <void (*bvecfunc)(const Vector&, Vector& )>
    void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda);
template <void (*bvecfunc)(const Vector&, Vector& )>
        void bbTTemplate(const Vector& xt, DenseMatrix& bbT);
template <void (*bvecfunc)(const Vector&, Vector& )>
    double bTbTemplate(const Vector& xt);
template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaNonHomoTemplate(const Vector& xt, Vector& sigma);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
    double rhsideTemplate(const Vector& xt);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void bfTemplate(const Vector& xt, Vector& bf);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        double divsigmaTemplate(const Vector& xt);

template<double (*S)(const Vector & xt) > double uNonhomoTemplate(const Vector& xt);


class Transport_test
    {
    protected:
        int dim;
        int numsol;

    public:
        FunctionCoefficient * scalaru;
        FunctionCoefficient * u_nonhomo;              // u_nonhomo(x,t) = u(x,t=0)
        FunctionCoefficient * scalarf;                // d (S - S_nonhomo) /dt + div (b [S - S_nonhomo]), Snonhomo = S(x,0)
        FunctionCoefficient * scalardivsigma;         // = dS/dt + div (bS) = div sigma
        FunctionCoefficient * bTb;
        VectorFunctionCoefficient * sigma;
        VectorFunctionCoefficient * conv;
        VectorFunctionCoefficient * bf;
        MatrixFunctionCoefficient * Ktilda;
        MatrixFunctionCoefficient * bbT;
        VectorFunctionCoefficient * sigma_nonhomo; // to incorporate inhomogeneous boundary conditions, stores (conv *S0, S0) with S(t=0) = S0
        FunctionCoefficientExtra  * weightedscalarf;
    public:
        Transport_test (int Dim, int NumSol);

        int GetDim() {return dim;}
        int GetNumSol() {return numsol;}
        void SetDim(int Dim) { dim = Dim;}
        void SetNumSol(int NumSol) { numsol = NumSol;}
        bool CheckTestConfig();

        ~Transport_test () {}
    private:
        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbvec)(const Vector & xt)> \
        void SetTestCoeffs ( );

        void SetScalarFun( double (*S)(const Vector & xt))
        { scalaru = new FunctionCoefficient(S);}

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        void SetScalarfFun()
        { scalarf = new FunctionCoefficient(rhsideTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>);}

        template< void(*f2)(const Vector & x, Vector & vec)>  \
        void SetScalarBtB()
        {
            bTb = new FunctionCoefficient(bTbTemplate<f2>);
        }

        template<double (*f1)(const Vector & xt), void(*f2)(const Vector & x, Vector & vec)> \
        void SetHdivVec()
        {
            sigma = new VectorFunctionCoefficient(dim, sigmaTemplate<f1,f2>);
        }

        void SetConvVec( void(*f)(const Vector & x, Vector & vec))
        { conv = new VectorFunctionCoefficient(dim, f);}

        template< void(*f2)(const Vector & x, Vector & vec)>  \
        void SetKtildaMat()
        {
            Ktilda = new MatrixFunctionCoefficient(dim, KtildaTemplate<f2>);
        }

        template< void(*bvec)(const Vector & x, Vector & vec)>  \
        void SetBBtMat()
        {
            bbT = new MatrixFunctionCoefficient(dim, bbTTemplate<bvec>);
        }

        template<double (*f1)(const Vector & xt), void(*f2)(const Vector & x, Vector & vec)> \
        void SetInitCondVec()
        {
            sigma_nonhomo = new VectorFunctionCoefficient(dim, sigmaNonHomoTemplate<f1,f2>);
        }

        template< double (*u)(const Vector & xt)>  \
        void SetuNonhomo()
        {
            u_nonhomo = new FunctionCoefficient(uNonhomoTemplate<u>);
        }

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        void SetConvfFunVec()
        { bf = new VectorFunctionCoefficient(dim, bfTemplate<S,dSdt,Sgradxvec,bvec,divbfunc>);}

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void SetDivSigma()
        { scalardivsigma = new FunctionCoefficient(divsigmaTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>);}

    };

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),  \
             void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
    void Transport_test::SetTestCoeffs ()
    {
        SetScalarFun(S);
        SetScalarfFun<S, dSdt, Sgradxvec, bvec, divbfunc>();
        SetConvVec(bvec);
        SetConvfFunVec<S, dSdt, Sgradxvec, bvec, divbfunc>();
        SetHdivVec<S,bvec>();
        SetKtildaMat<bvec>();
        SetScalarBtB<bvec>();
        SetInitCondVec<S,bvec>();
        SetuNonhomo<S>();
        SetDivSigma<S, dSdt, Sgradxvec, bvec, divbfunc>();
        SetBBtMat<bvec>();
        return;
    }


    bool Transport_test::CheckTestConfig()
    {
        if (dim == 4 || dim == 3)
        {
            if (numsol == 0)
                return true;
            if ( numsol == 1 && dim == 3 )
                return true;
            if ( numsol == 2 && dim == 4 )
                return true;
            if ( numsol == 3 && dim == 3 )
                return true;
            if ( numsol == 33 && dim == 4 )
                return true;
            if ( numsol == 4 && dim == 3 )
                return true;
            if ( numsol == 44 && dim == 3 )
                return true;
            if ( numsol == 100 && dim == 3 )
                return true;
            if ( numsol == 200 && dim == 3 )
                return true;
            if ( numsol == 5 && dim == 3 )
                return true;
            if ( numsol == 55 && dim == 4 )
                return true;
            if ( numsol == 444 && dim == 4 )
                return true;
            if ( numsol == 1000 && dim == 3 )
                return true;
            if ( numsol == 8 && dim == 3 )
                return true;
            if (numsol == 10 && dim == 4)
                return true;
            if (numsol == -3 && dim == 3)
                return true;
            if (numsol == -4 && dim == 4)
                return true;
            return false;
        }
        else
            return false;

    }

    Transport_test::Transport_test (int Dim, int NumSol)
    {
        dim = Dim;
        numsol = NumSol;

        if ( CheckTestConfig() == false )
            std::cout << "Inconsistent dim and numsol \n" << std::flush;
        else
        {
            if (numsol == -3) // 3D test for the paper
            {
                SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex>();
            }
            if (numsol == -4) // 4D test for the paper
            {
                SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex>();
            }
            if (numsol == 0)
            {
                //std::cout << "The domain is rectangular or cubic, velocity does not"
                             //" satisfy divergence condition" << std::endl << std::flush;
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex>();
                //SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex>();
            }
            if (numsol == 1)
            {
                //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
                SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_gradx, &bFunCircle2D_ex, &bFunCircle2Ddiv_ex>();
            }
            if (numsol == 100)
            {
                //std::cout << "The domain must be a cylinder over a unit square" << std::endl << std::flush;
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex>();
            }
            if (numsol == 200)
            {
                //std::cout << "The domain must be a cylinder over a unit circle" << std::endl << std::flush;
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunCircle2D_ex, &bFunCircle2Ddiv_ex>();
            }
            if (numsol == 2)
            {
                //std::cout << "The domain must be a cylinder over a 3D cube, velocity does not"
                             //" satisfy divergence condition" << std::endl << std::flush;
                SetTestCoeffs<&uFun3_ex, &uFun3_ex_dt, &uFun3_ex_gradx, &bFun_ex, &bFundiv_ex>();
            }
            if (numsol == 3)
            {
                //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunCircle2D_ex, &bFunCircle2Ddiv_ex>();
            }
            if (numsol == 4) // no exact solution in fact, ~ unsuccessfully trying to get a picture from the report
            {
                //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
                //std::cout << "Using new interface \n";
                SetTestCoeffs<&uFun5_ex, &uFun5_ex_dt, &uFun5_ex_gradx, &bFunCircle2D_ex, &bFunCircle2Ddiv_ex>();
            }
            if (numsol == 44) // no exact solution in fact, ~ unsuccessfully trying to get a picture from the report
            {
                //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
                //std::cout << "Using new interface \n";
                SetTestCoeffs<&uFun6_ex, &uFun6_ex_dt, &uFun6_ex_gradx, &bFunCircle2D_ex, &bFunCircle2Ddiv_ex>();
            }
            if (numsol == 8)
            {
                //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
                SetTestCoeffs<&uFunCylinder_ex, &uFunCylinder_ex_dt, &uFunCylinder_ex_gradx, &bFunCircle2D_ex, &bFunCircle2Ddiv_ex>();
            }
            if (numsol == 5)
            {
                //std::cout << "The domain must be a cylinder over a square" << std::endl << std::flush;
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex>();
            }
            if (numsol == 1000)
            {
                //std::cout << "The domain must be a cylinder over a square" << std::endl << std::flush;
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex>();
            }
            if (numsol == 33)
            {
                //std::cout << "The domain must be a cylinder over a sphere" << std::endl << std::flush;
                SetTestCoeffs<&uFun33_ex, &uFun33_ex_dt, &uFun33_ex_gradx, &bFunSphere3D_ex, &bFunSphere3Ddiv_ex>();
            }
            if (numsol == 444) // no exact solution in fact, ~ unsuccessfully trying to get something beauitiful
            {
                //std::cout << "The domain must be a cylinder over a sphere" << std::endl << std::flush;
                SetTestCoeffs<&uFun66_ex, &uFun66_ex_dt, &uFun66_ex_gradx, &bFunSphere3D_ex, &bFunSphere3Ddiv_ex>();
            }
            if (numsol == 55)
            {
                //std::cout << "The domain must be a cylinder over a cube" << std::endl << std::flush;
                SetTestCoeffs<&uFun33_ex, &uFun33_ex_dt, &uFun33_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex>();
            }
            if (numsol == 10)
            {
                SetTestCoeffs<&uFun10_ex, &uFun10_ex_dt, &uFun10_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex>();
            }
        } // end of setting test coefficients in correct case
    }

int main(int argc, char *argv[])
{
    int num_procs, myid;
    bool visualization = 0;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);

    int nDimensions     = 3;
    int numsol          = 0;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 3;

    /*
    int generate_frombase   = 0;
    int Nsteps              = 8;
    double tau              = 0.125;
    int generate_parallel   = generate_frombase * 1;
    int whichparallel       = generate_parallel * 2;
    int bnd_method          = 1;
    int local_method        = 2;
    */

    const char *formulation = "cfosls"; // "cfosls" or "fosls"

    // solver options
    int prec_option = 1; //defines whether to use preconditioner or not, and which one

    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    //const char *mesh_file = "../data/cube4d_low.MFEM";
    //const char *mesh_file = "../data/cube4d.MFEM";
    //const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../build3/pmesh_2_mwe_0.mesh";
    //const char *mesh_file = "../build3/mesh_par1_id0_np_1.mesh";
    //const char *mesh_file = "../build3/mesh_par1_id0_np_2.mesh";
    //const char *mesh_file = "../data/tempmesh_frompmesh.mesh";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";

    //const char * meshbase_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * meshbase_file = "../data/sphere3D_0.05to0.1.mesh";
    //const char * meshbase_file = "../data/sphere3D_veryfine.mesh";
    //const char * meshbase_file = "../data/beam-tet.mesh";
    //const char * meshbase_file = "../data/escher-p3.mesh";
    //const char * meshbase_file = "../data/orthotope3D_moderate.mesh";
    //const char * meshbase_file = "../data/orthotope3D_fine.mesh";
    //const char * meshbase_file = "../data/square_2d_moderate.mesh";
    //const char * meshbase_file = "../data/square_2d_fine.mesh";
    //const char * meshbase_file = "../data/square-disc.mesh";
    //const char *meshbase_file = "dsadsad";
    //const char * meshbase_file = "../data/circle_fine_0.1.mfem";
    //const char * meshbase_file = "../data/circle_moderate_0.2.mfem";

    int feorder         = 0;

    if (verbose)
        cout << "Solving (С)FOSLS Transport equation with MFEM & hypre" << endl;

    OptionsParser args(argc, argv);
    //args.AddOption(&mesh_file, "-m", "--mesh",
    //               "Mesh file to use.");
    //args.AddOption(&meshbase_file, "-mbase", "--meshbase",
    //               "Mesh base file to use.");
    args.AddOption(&feorder, "-o", "--feorder",
                   "Finite element order (polynomial degree).");
    args.AddOption(&ser_ref_levels, "-sref", "--sref",
                   "Number of serial refinements 4d mesh.");
    args.AddOption(&par_ref_levels, "-pref", "--pref",
                   "Number of parallel refinements 4d mesh.");
    args.AddOption(&nDimensions, "-dim", "--whichD",
                   "Dimension of the space-time problem.");
    /*
    args.AddOption(&Nsteps, "-nstps", "--nsteps",
                   "Number of time steps.");
    args.AddOption(&tau, "-tau", "--tau",
                   "Time step.");
    args.AddOption(&generate_frombase, "-gbase", "--genfrombase",
                   "Generating mesh from the base mesh.");
    args.AddOption(&generate_parallel, "-gp", "--genpar",
                   "Generating mesh in parallel.");
    args.AddOption(&whichparallel, "-pv", "--parver",
                   "Version of parallel algorithm.");
    args.AddOption(&bnd_method, "-bnd", "--bndmeth",
                   "Method for generating boundary elements.");
    args.AddOption(&local_method, "-loc", "--locmeth",
                   "Method for local mesh procedure.");
    args.AddOption(&numsol, "-nsol", "--numsol",
                   "Solution number.");
    */
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&prec_option, "-precopt", "--prec-option",
                   "Preconditioner choice.");
    args.AddOption(&formulation, "-form", "--formul",
                   "Formulation to use.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");

    args.Parse();
    if (!args.Good())
    {
       if (verbose)
       {
          args.PrintUsage(cout);
       }
       MPI_Finalize();
       return 1;
    }
    if (verbose)
    {
       args.PrintOptions(cout);
    }

    if (verbose)
        std::cout << "Running tests for the paper: \n";

    if (nDimensions == 3)
    {
        numsol = -3;
        mesh_file = "../data/cube_3d_moderate.mesh";
    }
    else // 4D case
    {
        numsol = -4;
        mesh_file = "../data/cube4d_96.MFEM";
    }

    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";



    if (verbose)
        cout << "Number of mpi processes: " << num_procs << endl << flush;

    StopWatch chrono;

    //DEFAULTED LINEAR SOLVER OPTIONS
    int max_iter = 150000;
    double rtol = 1e-12;//1e-7;//1e-9;
    double atol = 1e-14;//1e-9;//1e-12;

    Mesh *mesh = NULL;

    shared_ptr<ParMesh> pmesh;

    if (nDimensions == 3 || nDimensions == 4)
    {
        /*
        if ( generate_frombase == 1 )
        {
            if ( verbose )
                cout << "Creating a " << nDimensions << "d mesh from a " <<
                        nDimensions - 1 << "d mesh from the file " << meshbase_file << endl;

            Mesh * meshbase;
            ifstream imesh(meshbase_file);
            if (!imesh)
            {
                 cerr << "\nCan not open mesh file for base mesh: " <<
                                                    meshbase_file << endl << flush;
                 MPI_Finalize();
                 return -2;
            }
            meshbase = new Mesh(imesh, 1, 1);
            imesh.close();

            for (int l = 0; l < ser_ref_levels; l++)
                meshbase->UniformRefinement();

            if (verbose)
                meshbase->PrintInfo();

            if (generate_parallel == 1) //parallel version
            {
                ParMesh * pmeshbase = new ParMesh(comm, *meshbase);

                chrono.Clear();
                chrono.Start();

                if ( whichparallel == 1 )
                {
                    if ( nDimensions == 3)
                    {
                        if  (verbose)
                            cout << "Not implemented for 2D->3D. Use parallel version2"
                                    " instead" << endl << flush;
                        MPI_Finalize();
                        return 0;
                    }
                    else // nDimensions == 4
                    {
                        mesh = new Mesh( comm, *pmeshbase, tau, Nsteps, bnd_method, local_method);
                        if ( verbose )
                            cout << "Success: ParMesh is created by deprecated method"
                                 << endl << flush;

                        std::stringstream fname;
                        fname << "mesh_par1_id" << myid << "_np_" << num_procs << ".mesh";
                        std::ofstream ofid(fname.str().c_str());
                        ofid.precision(8);
                        mesh->Print(ofid);

                        MPI_Barrier(comm);
                    }
                }
                else
                {
                    if (verbose)
                        cout << "Starting parallel \"" << nDimensions-1 << "D->"
                             << nDimensions <<"D\" mesh generator" << endl;

                    pmesh = make_shared<ParMesh>( comm, *pmeshbase, tau, Nsteps,
                                                  bnd_method, local_method);

                    if (verbose)
                        cout << "Success: ParMesh created" << endl << flush;
                    MPI_Barrier(comm);
                }

                chrono.Stop();
                if (verbose && whichparallel == 2)
                    cout << "Timing: Space-time mesh extension done in parallel in "
                              << chrono.RealTime() << " seconds.\n" << endl << flush;
                delete pmeshbase;
            }
            else // serial version
            {
                if (verbose)
                    cout << "Starting serial \"" << nDimensions-1 << "D->"
                         << nDimensions <<"D\" mesh generator" << endl;
                mesh = new Mesh( *meshbase, tau, Nsteps, bnd_method, local_method);
                if (verbose)
                    cout << "Timing: Space-time mesh extension done in serial in "
                              << chrono.RealTime() << " seconds.\n" << endl << flush;
            }

            delete meshbase;

        }
        else
        */ // not generating from a lower dimensional mesh
        {
            if (verbose)
                cout << "Reading a " << nDimensions << "d mesh from the file " << mesh_file << endl;
            ifstream imesh(mesh_file);
            if (!imesh)
            {
                 std::cerr << "\nCan not open mesh file: " << mesh_file << '\n' << std::endl;
                 MPI_Finalize();
                 return -2;
            }
            else
            {
                mesh = new Mesh(imesh, 1, 1);
                imesh.close();
            }

        }

    }
    else //if nDimensions is no 3 or 4
    {
        if (verbose)
            cerr << "Case nDimensions = " << nDimensions << " is not supported \n"
                 << flush;
        MPI_Finalize();
        return -1;

    }

    if (mesh) // if only serial mesh was generated previously, parallel mesh is initialized here
    {
        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        if ( verbose )
            cout << "Creating parmesh(" << nDimensions <<
                    "d) from the serial mesh (" << nDimensions << "d)" << endl << flush;
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }

    for (int l = 0; l < par_ref_levels; l++)
    {
       pmesh->UniformRefinement();
    }

    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.
    int dim = nDimensions;

    FiniteElementCollection *hdiv_coll;
    if ( dim == 4 )
    {
        hdiv_coll = new RT0_4DFECollection;
        if(verbose)
            cout << "RT: order 0 for 4D" << endl;
    }
    else
    {
        hdiv_coll = new RT_FECollection(feorder, dim);
        if(verbose)
            cout << "RT: order " << feorder << " for 3D" << endl;
    }

    if (dim == 4)
        MFEM_ASSERT(feorder==0, "Only lowest order elements are support in 4D!");
    FiniteElementCollection *h1_coll;
    if (dim == 4)
    {
        h1_coll = new LinearFECollection;
        if (verbose)
            cout << "H1 in 4D: linear elements are used" << endl;
    }
    else
    {
        h1_coll = new H1_FECollection(feorder+1, dim);
        if(verbose)
            cout << "H1: order " << feorder + 1 << " for 3D" << endl;
    }
    FiniteElementCollection *l2_coll = new L2_FECollection(feorder, dim);
    if(verbose)
        cout << "L2: order " << feorder << endl;

    ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);
    ParFiniteElementSpace *H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);
    ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimH = H_space->GlobalTrueVSize();
    HYPRE_Int dimW = W_space->GlobalTrueVSize();

    if (verbose)
    {
       std::cout << "***********************************************************\n";
       std::cout << "dim(R) = " << dimR << ", ";
       std::cout << "dim(H) = " << dimH << ", ";
       if (strcmp(formulation,"cfosls") == 0)
       {
            std::cout << "dim(W) = " << dimW << ", ";
            std::cout << "dim(R+H+W) = " << dimR + dimH + dimW << "\n";
       }
       else // fosls
           std::cout << "dim(R+H) = " << dimR + dimH << "\n";
       std::cout << "***********************************************************\n";
    }

    // 7. Define the two BlockStructure of the problem.  block_offsets is used
    //    for Vector based on dof (like ParGridFunction or ParLinearForm),
    //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
    //    for the rhs and solution of the linear system).  The offsets computed
    //    here are local to the processor.
    int numblocks = 2;
    if (strcmp(formulation,"cfosls") == 0)
        numblocks = 3;

    Array<int> block_offsets(numblocks + 1); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = R_space->GetVSize();
    block_offsets[2] = H_space->GetVSize();
    if (strcmp(formulation,"cfosls") == 0)
        block_offsets[3] = W_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(numblocks + 1); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = R_space->TrueVSize();
    block_trueOffsets[2] = H_space->TrueVSize();
    if (strcmp(formulation,"cfosls") == 0)
        block_trueOffsets[3] = W_space->TrueVSize();
    block_trueOffsets.PartialSum();

    BlockVector x(block_offsets), rhs(block_offsets);
    BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
    x = 0.0;
    trueX = 0.0;
    trueRhs = 0.0;

   // 8. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(.0);

   Transport_test Mytest(nDimensions,numsol);

   //----------------------------------------------------------
   // Setting boundary conditions.
   //----------------------------------------------------------

   // for sigma essential bdr's are in=mposed everywhere except top t = t_final
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   ess_bdr[pmesh->bdr_attributes.Max()-1] = 0;
   //ess_bdr = 0;
   //ess_bdr[0] = 1; // t = 0
   //ess_bdr[1] = 1; // lateral boundary in case of 3 bdr attributes

   // for S boundary conditions are essential only for t = 0 in parabolic case
   Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
   ess_bdrS = 0;
   ess_bdrS[0] = 1; // t = 0
   //ess_bdr[1] = 1; // lateral boundary
   //R_space->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   //-----------------------

   // 9. Define the parallel grid function and parallel linear forms, solution
   //    vector and rhs.

   ParLinearForm *fform(new ParLinearForm);
   fform->Update(R_space, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectordivDomainLFIntegrator(zero));
//   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(gradfcoeff));
//   fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fcoeff));
   fform->Assemble();
//   fform->ParallelAssemble(trueRhs.GetBlock(0));

   ParLinearForm *qform(new ParLinearForm);
   qform->Update(H_space, rhs.GetBlock(1), 0);
//   qform->AddDomainIntegrator(new DomainLFIntegrator(zero));
   //qform->AddDomainIntegrator(new GradDomainLFIntegrator(bfFun));
   qform->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bf));
   qform->Assemble();//qform->Print();
//   qform->ParallelAssemble(trueRhs.GetBlock(1));

   ParLinearForm *gform(new ParLinearForm);
   if (strcmp(formulation,"cfosls") == 0)
   {
       gform->Update(W_space, rhs.GetBlock(2), 0);
       //gform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
       gform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalarf));
       gform->Assemble();
       //gform->ParallelAssemble(trueRhs.GetBlock(2));
   }

   // 10. Assemble the finite element matrices for the CFOSLS operator  A
   //     where:

   ParBilinearForm *Ablock(new ParBilinearForm(R_space));
   HypreParMatrix *A;
//    Ablock->AddDomainIntegrator(new DivDivIntegrator());
//    Ablock->AddDomainIntegrator(new VectorFEMassIntegrator(mass_k));
   Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
   Ablock->Assemble();
   Ablock->EliminateEssentialBC(ess_bdr,x.GetBlock(0),*fform);
   Ablock->Finalize();
   A = Ablock->ParallelAssemble();

//---------------
//  C Block:
//---------------

   //MatrixFunctionCoefficient bbT( dim, bbT_ex );
//    VectorFEMassIntegrator bbT(bbT_coef);

   ParBilinearForm *Cblock(new ParBilinearForm(H_space));
   HypreParMatrix *C;
   //Cblock->AddDomainIntegrator(new MassIntegrator(bTb));
   //Cblock->AddDomainIntegrator(new DiffusionIntegrator(bbT));
   Cblock->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));
   Cblock->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.bbT));
   Cblock->Assemble();
   Cblock->EliminateEssentialBC(ess_bdrS, x.GetBlock(1),*qform);
   Cblock->Finalize();
   C = Cblock->ParallelAssemble();

//---------------
//  B Block:
//---------------

   Vector et(dim); et = 0.; et(dim-1) = 1.0;
   VectorConstantCoefficient bt(et);
   ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(R_space, H_space));
   HypreParMatrix *B;
   //Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(bFun));
   Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.conv));
//    Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(bt));
   Bblock->Assemble();
   Bblock->EliminateTrialDofs(ess_bdr, x.GetBlock(0), *qform);
   Bblock->EliminateTestDofs(ess_bdrS);
   Bblock->Finalize();
   B = Bblock->ParallelAssemble();
   *B *= -1.;
   HypreParMatrix *BT = B->Transpose();

//----------------
//  D Block:
//-----------------
   HypreParMatrix *D;
   HypreParMatrix *DT;

   if (strcmp(formulation,"cfosls") == 0)
   {
      ParMixedBilinearForm *Dblock(new ParMixedBilinearForm(R_space, W_space));
      Dblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
      Dblock->Assemble();
      Dblock->EliminateTrialDofs(ess_bdr, x.GetBlock(0), *gform);;
      Dblock->Finalize();
      D = Dblock->ParallelAssemble();
      DT = D->Transpose();
   }

//=======================================================
// Assembling the Matrix
//-------------------------------------------------------

  fform->ParallelAssemble(trueRhs.GetBlock(0));
  qform->ParallelAssemble(trueRhs.GetBlock(1));
  if (strcmp(formulation,"cfosls") == 0)
     gform->ParallelAssemble(trueRhs.GetBlock(2));
  BlockOperator *CFOSLSop = new BlockOperator(block_trueOffsets);
  CFOSLSop->SetBlock(0,0, A);
  CFOSLSop->SetBlock(0,1, BT);
  CFOSLSop->SetBlock(1,0, B);
  CFOSLSop->SetBlock(1,1, C);
  if (strcmp(formulation,"cfosls") == 0)
  {
    CFOSLSop->SetBlock(0,2, DT);
    CFOSLSop->SetBlock(2,0, D);
  }

   if (verbose)
       cout<< "Final saddle point matrix assembled"<<endl << flush;
   MPI_Barrier(MPI_COMM_WORLD);

   // 12. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.

   // Construct the operators for preconditioner
   if (verbose)
       cout << "Using Diag(A) + BoomerAMG(C) + BoomerAMG(D Diag^(-1)(A) D^t) as a preconditioner" << endl;
   chrono.Clear();
   chrono.Start();

   HypreParMatrix *Schur;
   if (strcmp(formulation,"cfosls") == 0 )
   {
      HypreParMatrix *AinvDt = D->Transpose();
      HypreParVector *Ad = new HypreParVector(MPI_COMM_WORLD, A->GetGlobalNumRows(),
                                           A->GetRowStarts());
      A->GetDiag(*Ad);
      AinvDt->InvScaleRows(*Ad);
      Schur = ParMult(D, AinvDt);
   }

   HypreDiagScale * invA = new HypreDiagScale(*A);

   HypreBoomerAMG * invC = new HypreBoomerAMG(*C);
   invC->SetPrintLevel(0);

   Solver * invS;
   if (strcmp(formulation,"cfosls") == 0 )
   {
        invS = new HypreBoomerAMG(*Schur);
        ((HypreBoomerAMG *)invS)->SetPrintLevel(0);
        ((HypreBoomerAMG *)invS)->iterative_mode = false;
   }

   invA->iterative_mode = false;
   invC->iterative_mode = false;

   BlockDiagonalPreconditioner prec(block_trueOffsets);
   if (prec_option > 0)
   {
       prec.SetDiagonalBlock(0, invA);
       prec.SetDiagonalBlock(1, invC);
       if (strcmp(formulation,"cfosls") == 0)
            prec.SetDiagonalBlock(2, invS);

       if (verbose)
           std::cout << "Preconditioner built in " << chrono.RealTime() << "s. \n";
   }
   else
       if (verbose)
           cout << "No preconditioner is used." << endl;

   // 12. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.

   chrono.Clear();
   chrono.Start();
   MINRESSolver solver(MPI_COMM_WORLD);
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(max_iter);
   solver.SetOperator(*CFOSLSop);
   if (prec_option > 0)
        solver.SetPreconditioner(prec);
   solver.SetPrintLevel(0);
   trueX = 0.0;
   solver.Mult(trueRhs, trueX);
   chrono.Stop();

   if (verbose)
   {
      if (solver.GetConverged())
         std::cout << "MINRES converged in " << solver.GetNumIterations()
                   << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
      else
         std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                   << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
      std::cout << "MINRES solver took " << chrono.RealTime() << "s. \n";
   }

   ParGridFunction * sigma = new ParGridFunction(R_space);
   sigma->Distribute(&(trueX.GetBlock(0)));

   ParGridFunction * S = new ParGridFunction(H_space);
   S->Distribute(&(trueX.GetBlock(1)));

   ParGridFunction *sigma_exact = new ParGridFunction(R_space);
   //sigma_exact->ProjectCoefficient(sigmacoeff);
   sigma_exact->ProjectCoefficient(*(Mytest.sigma));
   HypreParVector * sigma_exactpv = sigma_exact->ParallelAssemble();
   Vector * sigma_exactv = sigma_exactpv->GlobalVector();

   ParGridFunction *S_exact = new ParGridFunction(H_space);
   S_exact->ProjectCoefficient(*(Mytest.scalaru));
   HypreParVector * S_exactpv = S_exact->ParallelAssemble();
   Vector * S_exactv = S_exactpv->GlobalVector();


   // adding back the term from nonhomogeneous initial condition
   ParGridFunction *sigma_nonhomo = new ParGridFunction(R_space);
   sigma_nonhomo->ProjectCoefficient(*(Mytest.sigma_nonhomo));
   *sigma += *sigma_nonhomo;

   ParGridFunction *S_nonhomo = new ParGridFunction(H_space);
   S_nonhomo->ProjectCoefficient(*(Mytest.u_nonhomo));
   *S += *S_nonhomo;

   // 13. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor. Compute
   //     L2 error norms.

   int order_quad = max(2, 2*feorder+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }


   double err_sigma = sigma->ComputeL2Error(*(Mytest.sigma), irs);
   double norm_sigma = ComputeGlobalLpNorm(2, *(Mytest.sigma), *pmesh, irs);
   if (verbose)
       cout << "|| sigma - sigma_ex || / || sigma_ex || = " << err_sigma / norm_sigma << endl;
       //cout << "local: err_sigma / norm_sigma = " << err_sigma / norm_sigma << endl;

   /*
   err_sigma *= err_sigma;
   double err_sigma_global;
   MPI_Reduce(&err_sigma, &err_sigma_global, 1, MPI_DOUBLE,
              MPI_SUM, 0, comm);
   err_sigma_global = std::sqrt(err_sigma_global);

   norm_sigma *= norm_sigma;
   double norm_sigma_global;
   MPI_Reduce(&norm_sigma, &norm_sigma_global, 1, MPI_DOUBLE,
              MPI_SUM, 0, comm);
   norm_sigma_global = std::sqrt(norm_sigma_global);

   if (verbose)
       cout << "global: err_sigma / norm_sigma = " << err_sigma_global / norm_sigma_global << endl;
   */

   DiscreteLinearOperator Div(R_space, W_space);
   Div.AddDomainInterpolator(new DivergenceInterpolator());
   ParGridFunction DivSigma(W_space);
   Div.Assemble();
   Div.Mult(*sigma, DivSigma);

   double err_div = DivSigma.ComputeL2Error(*(Mytest.scalardivsigma),irs);
   double norm_div = ComputeGlobalLpNorm(2, *(Mytest.scalardivsigma), *pmesh, irs);

   if (verbose)
   {
       cout << "|| div (sigma_h - sigma_ex) || / ||div (sigma_ex)|| = "
                 << err_div/norm_div  << "\n";
   }

   if (verbose)
   {
       cout << "Actually it will be ~ continuous L2 + discrete L2 for divergence" << endl;
       cout << "|| sigma_h - sigma_ex ||_Hdiv / || sigma_ex ||_Hdiv = "
                 << sqrt(err_sigma*err_sigma + err_div * err_div)/sqrt(norm_sigma*norm_sigma + norm_div * norm_div)  << "\n";
   }


   // Computing error for S

   double err_S = S->ComputeL2Error((*Mytest.scalaru), irs);
   double norm_S = ComputeGlobalLpNorm(2, (*Mytest.scalaru), *pmesh, irs);
   if (verbose)
   {
       std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                    err_S / norm_S << "\n";
   }

   /*
   if (verbose)
       cout << "Computing mesh norms" << endl;

   HypreParVector * sigmapv = sigma->ParallelAssemble();
   Vector * sigmav = sigmapv->GlobalVector();
   *sigmav -= *sigma_exactv;

   double sigma_meshnorm = (*sigma_exactv)*(*sigma_exactv);
   double sigma_mesherror = (*sigmav) * (*sigmav);
   if(verbose)
       cout << "|| sigma_h - sigma_ex ||_h / || sigma_ex ||_h = "
                       << sqrt(sigma_mesherror) / sqrt(sigma_meshnorm) << endl;

   BilinearForm *m = new BilinearForm(R_space);
   m->AddDomainIntegrator(new DivDivIntegrator);
   m->AddDomainIntegrator(new VectorFEMassIntegrator);
   m->Assemble(); m->Finalize();
   SparseMatrix E = m->SpMat();
   Vector Asigma(sigmav->Size());
   E.Mult(*sigma_exactv,Asigma);
   double weighted_norm = (*sigma_exactv)*Asigma;

   Vector Ae(sigmav->Size());
   E.Mult(*sigmav,Ae);
   double weighted_error = (*sigmav)*Ae;

   if(verbose)
       cout << "|| sigma_h - sigma_ex ||_h,Hdiv / || sigma_ex ||_h,Hdiv = " <<
                       sqrt(weighted_error)/sqrt(weighted_norm) << endl;


   HypreParVector * Spv = S->ParallelAssemble();
   Vector * Sv = Spv->GlobalVector();
   *Sv -= *S_exactv;

   double S_meshnorm = (*S_exactv)*(*S_exactv);
   double S_mesherror = (*Sv) * (*Sv);
   if(verbose)
       cout << "|| S_h - S_ex ||_h / || S_ex ||_h = "
                       << sqrt(S_mesherror) / sqrt(S_meshnorm) << endl;
    */
   if (verbose)
       cout << "Computing projection errors" << endl;

   double projection_error_sigma = sigma_exact->ComputeL2Error(*(Mytest.sigma), irs);

   if(verbose)
   {
       cout << "|| sigma_ex - Pi_h sigma_ex || / || sigma_ex || = "
                       << projection_error_sigma / norm_sigma << endl;
   }

   double projection_error_S = S_exact->ComputeL2Error(*(Mytest.scalaru), irs);

   if(verbose)
       cout << "|| S_ex - Pi_h S_ex || / || S_ex || = "
                       << projection_error_S / norm_S << endl;


   if (visualization && nDimensions < 4)
   //if (true)
   {
      //cout << "visualization may not work for 4D element code and not present in mfem_4d version" << endl;
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock << "parallel " << num_procs << " " << myid << "\n";
      u_sock.precision(8);
      u_sock << "solution\n" << *pmesh << *sigma_exact << "window_title 'sigma_exact'"
             << endl;
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):


      socketstream uu_sock(vishost, visport);
      uu_sock << "parallel " << num_procs << " " << myid << "\n";
      uu_sock.precision(8);
      uu_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma'"
             << endl;

      *sigma_exact -= * sigma_nonhomo;
      socketstream uuuuu_sock(vishost, visport);
      uuuuu_sock << "parallel " << num_procs << " " << myid << "\n";
      uuuuu_sock.precision(8);
      uuuuu_sock << "solution\n" << *pmesh << *sigma_exact << "window_title 'sigma - sigmanonhomo'"
             << endl;

      *sigma_exact += *sigma_nonhomo;

      socketstream uuuu_sock(vishost, visport);
      uuuu_sock << "parallel " << num_procs << " " << myid << "\n";
      uuuu_sock.precision(8);
      uuuu_sock << "solution\n" << *pmesh << *sigma_nonhomo << "window_title 'sigma_nonhomo'"
             << endl;

      *sigma_exact -= *sigma;

      socketstream uuu_sock(vishost, visport);
      uuu_sock << "parallel " << num_procs << " " << myid << "\n";
      uuu_sock.precision(8);
      uuu_sock << "solution\n" << *pmesh << *sigma_exact << "window_title 'difference'"
             << endl;

      MPI_Barrier(pmesh->GetComm());
   }

   // 17. Free the used memory.
   //delete fform;
   //delete CFOSLSop;
   //delete A;

   //delete Ablock;
   delete W_space;
   delete R_space;
   delete l2_coll;
   delete h1_coll;
   delete hdiv_coll;

   //delete pmesh;

   MPI_Finalize();

   return 0;
}

template <void (*bvecfunc)(const Vector&, Vector& )> \
void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda)
{
    int nDimensions = xt.Size();
    Vector b;
    bvecfunc(xt,b);
    double bTbInv = (-1./(b*b));
    Ktilda.Diag(1.0,nDimensions);
    AddMult_a_VVt(bTbInv,b,Ktilda);
}

template <void (*bvecfunc)(const Vector&, Vector& )> \
void bbTTemplate(const Vector& xt, DenseMatrix& bbT)
{
    int nDimensions = xt.Size();
    Vector b;
    bvecfunc(xt,b);
    MultVVt(b, bbT);
}

template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )> \
void sigmaTemplate(const Vector& xt, Vector& sigma)
{
    Vector b;
    bvecfunc(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = ufunc(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);

    return;
}

template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )> \
void sigmaNonHomoTemplate(const Vector& xt, Vector& sigma) // sigmaNonHomo = (b u, u) for u = S(t=0)
{
    Vector xteq0(xt.Size()); // xt with t = 0
    xteq0 = xt;
    xteq0(xteq0.Size()-1) = 0.0;

    Vector b;
    bvecfunc(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = ufunc(xteq0);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);

    return;
}


template <void (*bvecfunc)(const Vector&, Vector& )> \
double bTbTemplate(const Vector& xt)
{
    Vector b;
    bvecfunc(xt,b);
    return b*b;
}


template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
double divsigmaTemplate(const Vector& xt)
{
    Vector b;
    bvec(xt,b);

    Vector gradS;
    Sgradxvec(xt,gradS);

    double res = 0.0;

    res += dSdt(xt);
    for ( int i= 0; i < xt.Size() - 1; ++i )
        res += b(i) * gradS(i);
    res += divbfunc(xt) * S(xt);

    return res;
}

template<double (*S)(const Vector & xt) > double uNonhomoTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return S(xt0);
}


template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
double rhsideTemplate(const Vector& xt)
{
    Vector b;
    bvec(xt,b);

    Vector gradS;
    Sgradxvec(xt,gradS);

    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    Vector gradS0;
    Sgradxvec(xt0,gradS0);

    double res = 0.0;

    res += dSdt(xt);
    for ( int i= 0; i < xt.Size() - 1; ++i )
        res += b(i) * (gradS(i) - gradS0(i));
    res += divbfunc(xt) * (S(xt) - S(xt0));

    // only for debugging casuality weight usage
    //double t = xt[xt.Size()-1];
    //res *= exp (-t / 0.01);

    return res;

    /*
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    Vector b(3);
    bFunCircle2D_ex(xt,b);
    return 0.0 - (
           -100.0 * 2.0 * (x-0.5) * exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y)) * b(0) +
           -100.0 * 2.0 *    y    * exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y)) * b(1) );
    */
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
void bfTemplate(const Vector& xt, Vector& bf)
{
    Vector b;
    bvec(xt,b);

    Vector gradS;
    Sgradxvec(xt,gradS);

    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    Vector gradS0;
    Sgradxvec(xt0,gradS0);

    double res = 0.0;

    res += dSdt(xt);
    for ( int i= 0; i < xt.Size() - 1; ++i )
        res += b(i) * (gradS(i) - gradS0(i));
    res += divbfunc(xt) * (S(xt) - S(xt0));

    bf.SetSize(xt.Size());

    for (int i = 0; i < bf.Size(); ++i)
        bf(i) = res * b(i);
}



double uFun_ex(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    //return t;
    ////setback
    return sin(t)*exp(t);
}

double uFun_ex_dt(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    return (cos(t) + sin(t)) * exp(t);
}

void uFun_ex_gradx(const Vector& xt, Vector& gradx )
{
    gradx.SetSize(xt.Size() - 1);
    gradx = 0.0;
}


/*

double fFun(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    //double tmp = (xt.Size()==4) ? 1.0 - 2.0 * xt(2) : 0;
    double tmp = (xt.Size()==4) ? 2*M_PI * sin(2*xt(2)*M_PI) : 0;
    //double tmp = (xt.Size()==4) ? M_PI * cos(xt(2)*M_PI) : 0;
    //double tmp = (xt.Size()==4) ? M_PI * sin(xt(2)*M_PI) : 0;
    return cos(t)*exp(t)+sin(t)*exp(t)+(M_PI*cos(xt(1)*M_PI)*cos(xt(0)*M_PI)+
                   2*M_PI*cos(xt(0)*2*M_PI)*cos(xt(1)*M_PI)+tmp) *uFun_ex(xt);
    //return cos(t)*exp(t)+sin(t)*exp(t)+(1.0 - 2.0 * xt(0) + 1.0 - 2.0 * xt(1) +tmp) *uFun_ex(xt);
}
*/

void bFun_ex(const Vector& xt, Vector& b )
{
    b.SetSize(xt.Size());

    //for (int i = 0; i < xt.Size()-1; i++)
        //b(i) = xt(i) * (1 - xt(i));

    //if (xt.Size() == 4)
        //b(2) = 1-cos(2*xt(2)*M_PI);
        //b(2) = sin(xt(2)*M_PI);
        //b(2) = 1-cos(xt(2)*M_PI);

    b(0) = sin(xt(0)*2*M_PI)*cos(xt(1)*M_PI);
    b(1) = sin(xt(1)*M_PI)*cos(xt(0)*M_PI);
    b(2) = 1-cos(2*xt(2)*M_PI);

    b(xt.Size()-1) = 1.;

}

double bFundiv_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    if (xt.Size() == 4)
        return 2*M_PI * cos(x*2*M_PI)*cos(y*M_PI) + M_PI * cos(y*M_PI)*cos(x*M_PI) + 2*M_PI * sin(2*z*M_PI);
    if (xt.Size() == 3)
        return 2*M_PI * cos(x*2*M_PI)*cos(y*M_PI) + M_PI * cos(y*M_PI)*cos(x*M_PI);
    return 0.0;
}

double uFun2_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    return t * exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y));
}

double uFun2_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return (1.0 + t) * exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y));
}

void uFun2_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = t * exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y));
    gradx(1) = t * exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y));
}

/*
double fFun2(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    Vector b(3);
    bFunCircle2D_ex(xt,b);
    return (t + 1) * exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y)) +
             t * exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y)) * b(0) +
             t * exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y)) * b(1);
}
*/

double uFun3_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    return sin(t)*exp(t) * sin ( M_PI * (x + y + z));
}

double uFun3_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    return (sin(t) + cos(t)) * exp(t) * sin ( M_PI * (x + y + z));
}

void uFun3_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = sin(t) * exp(t) * M_PI * cos ( M_PI * (x + y + z));
    gradx(1) = sin(t) * exp(t) * M_PI * cos ( M_PI * (x + y + z));
    gradx(2) = sin(t) * exp(t) * M_PI * cos ( M_PI * (x + y + z));
}


/*
double fFun3(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    Vector b(4);
    bFun_ex(xt,b);

    return (cos(t)*exp(t)+sin(t)*exp(t)) * sin ( M_PI * (x + y + z)) +
            sin(t)*exp(t) * M_PI * cos ( M_PI * (x + y + z)) * b(0) +
            sin(t)*exp(t) * M_PI * cos ( M_PI * (x + y + z)) * b(1) +
            sin(t)*exp(t) * M_PI * cos ( M_PI * (x + y + z)) * b(2) +
            (2*M_PI*cos(x*2*M_PI)*cos(y*M_PI) +
             M_PI*cos(y*M_PI)*cos(x*M_PI)+
             + 2*M_PI*sin(z*2*M_PI)) * uFun3_ex(xt);
}
*/

double uFun4_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    return exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y));
}

double uFun4_ex_dt(const Vector& xt)
{
    return uFun4_ex(xt);
}

void uFun4_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y));
    gradx(1) = exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y));
}

double uFun33_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    return exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y + (z -0.25)*(z-0.25) ));
}

double uFun33_ex_dt(const Vector& xt)
{
    return uFun33_ex(xt);
}

void uFun33_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y + (z -0.25)*(z-0.25)));
    gradx(1) = exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y + (z -0.25)*(z-0.25)));
    gradx(2) = exp(t) * 2.0 * (z -0.25) * cos (((x - 0.5)*(x - 0.5) + y*y + (z -0.25)*(z-0.25)));
}
/*
double fFun4(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    Vector b(3);
    bFunCircle2D_ex(xt,b);
    return exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y)) +
             exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y)) * b(0) +
             exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y)) * b(1);
}


double f_natural(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    if ( t > MYZEROTOL)
        return 0.0;
    else
        return (-uFun5_ex(xt));
}
*/

double uFun5_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    if ( t < MYZEROTOL)
        return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y));
    else
        return 0.0;
}

double uFun5_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return 0.0;
}

void uFun5_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = -100.0 * 2.0 * (x - 0.5) * uFun5_ex(xt);
    gradx(1) = -100.0 * 2.0 * y * uFun5_ex(xt);
}


double uFun6_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y)) * exp(-10.0*t);
}

double uFun6_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return -10.0 * uFun6_ex(xt);
}

void uFun6_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = -100.0 * 2.0 * (x - 0.5) * uFun6_ex(xt);
    gradx(1) = -100.0 * 2.0 * y * uFun6_ex(xt);
}


double GaussianHill(const Vector&xvec)
{
    double x = xvec(0);
    double y = xvec(1);
    return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y));
}

double uFunCylinder_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double r = sqrt(x*x + y*y);
    double teta = atan(y/x);
    /*
    if (fabs(x) < MYZEROTOL && y > 0)
        teta = M_PI / 2.0;
    else if (fabs(x) < MYZEROTOL && y < 0)
        teta = - M_PI / 2.0;
    else
        teta = atan(y,x);
    */
    double t = xt(xt.Size()-1);
    Vector xvec(2);
    xvec(0) = r * cos (teta - t);
    xvec(1) = r * sin (teta - t);
    return GaussianHill(xvec);
}

double uFunCylinder_ex_dt(const Vector& xt)
{
    return 0.0;
}

void uFunCylinder_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = 0.0;
    gradx(1) = 0.0;
}


double uFun66_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y + (z - 0.25)*(z - 0.25))) * exp(-10.0*t);
}

double uFun66_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    return -10.0 * uFun6_ex(xt);
}

void uFun66_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = -100.0 * 2.0 * (x - 0.5)  * uFun6_ex(xt);
    gradx(1) = -100.0 * 2.0 * y          * uFun6_ex(xt);
    gradx(2) = -100.0 * 2.0 * (z - 0.25) * uFun6_ex(xt);
}

void Hdivtest_fun(const Vector& xt, Vector& out )
{
    out.SetSize(xt.Size());

    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    out(0) = x;
    out(1) = 0.0;
    out(2) = 0.0;
    out(xt.Size()-1) = 0.;

}

double L2test_fun(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    return x;
}


double uFun10_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return sin(t)*exp(t)*x*y;
}

double uFun10_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    return (cos(t)*exp(t) + sin(t)*exp(t)) * x * y;
}

void uFun10_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = sin(t)*exp(t)*y;
    gradx(1) = sin(t)*exp(t)*x;
    gradx(2) = 0.0;
}

