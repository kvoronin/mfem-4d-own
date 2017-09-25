//                                MFEM (with 4D elements) CFOSLS combined with Petrov-Galerkin formulation
//                                  using S from H1 for 3D/4D hyperbolic equation
//
// Compile with: make
//
// Description:  This example code solves a simple 3D/4D hyperbolic problem over [0,1]^{d+1}, d = 3 or 4
//               corresponding to the saddle point system
//                                  sigma - [b, 1]^T * S = 0
//                                  div_(x,t) sigma      = f
//                       where b is a given vector function (aka velocity),
//						 NO boundary conditions (which work only in case when b * n = 0 pointwise at the domain space boundary)
//						 and initial condition:
//                                  u(x,0)            = 0
//                       The system is solved via minimization principle:
//                                  J(sigma,S) = || sigma ||_H(div)^2 + || S ||_H1^2 ----> min
//                       over H(div) x H1, with Petrov-Galerkin constraints
//                                  (sigma - [b, 1]^T * S, bteta) = 0, bteta from L2^{d+1}
//                                  (div_(x,t) sigma - f, teta)   = 0, teta from L2
//                       Lambda and mu are the corresponding Lagrange multipliers.
//               For tests we use a manufactured exact solution and compute the corresponding r.h.s.
//               We discretize with Raviart-Thomas finite elements (sigma), continuous H1 elements (S) and
//					  discontinuous vector polynomials (lambda) and discontinuous polynomials (mu).
//
// Solver: MINRES with a block-diagonal preconditioner (using BoomerAMG)

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

#include"cfosls_testsuite.hpp"
#include "divfree_solver_tools.hpp"

//#define DEBUGGING

#define MYZEROTOL (1.0e-13)

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;


// Some bilinear and linear form integrators used in the code

/// Integrator for (q * div u, div v)
/// where q is a scalar coefficient, u and v are from vector FE space
/// created from scalar FE collection (called improper vector FE)
class ImproperDivDivIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient *Q;
   void Init(Coefficient *q)
   { Q = q; }

#ifndef MFEM_THREAD_SAFE
   Vector scalar_shape;
   DenseMatrix dshape; // components are test shapes
   DenseMatrix gshape;
   DenseMatrix Jadj;
   Vector div_shape;
#endif

public:
   ImproperDivDivIntegrator() { Init(NULL); }
   ImproperDivDivIntegrator(Coefficient *_q) { Init(_q); }
   ImproperDivDivIntegrator(Coefficient &q) { Init(&q); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

void ImproperDivDivIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
    MFEM_ASSERT(el.GetRangeType() == FiniteElement::SCALAR,
                "The improper vector FE should have a scalar type in the current implementation \n");

    int dim  = el.GetDim();
    int nd = el.GetDof();
    int improper_nd = nd * dim;

    double w;

#ifdef MFEM_THREAD_SAFE
    Vector scalar_shape(nd);
    DenseMatrix dshape(nd, dim);
    DenseMatrix gshape(nd, dim);
    DenseMatrix Jadj(dim);
    Vector div_shape(improper_nd);
#else
    scalar_shape.SetSize(nd);
    dshape.SetSize(nd, dim);
    gshape.SetSize (nd, dim);
    Jadj.SetSize (dim);
    div_shape.SetSize(improper_nd);
#endif
    elmat.SetSize(improper_nd);

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

       Trans.SetIntPoint (&ip);

       el.CalcDShape(ip, dshape);

       // taken from VectorDivergenceIntegrator as an older example of
       // handling improper vector f.e. divergence
       // but there was only one gradient, so inverse instead of adjuagte + det. here
       CalcInverse(Trans.Jacobian(), Jadj);
       Mult (dshape, Jadj, gshape);
       gshape.GradToDiv (div_shape);

       w = ip.weight * Trans.Weight();
       if (Q)
          w *= Q->Eval(Trans, ip);

       AddMult_a_VVt (w, div_shape, elmat);
    }
}

/// Integrator for (q * u, v)
/// where q is a scalar coefficient, u and v are from vector FE space
/// created from scalar FE collection (called improper vector FE)
class ImproperVectorFEMassIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient *Q;
   void Init(Coefficient *q)
   { Q = q; }

#ifndef MFEM_THREAD_SAFE
   Vector scalar_shape;
   DenseMatrix vector_vshape; // components are test shapes
#endif

public:
   ImproperVectorFEMassIntegrator() { Init(NULL); }
   ImproperVectorFEMassIntegrator(Coefficient *_q) { Init(_q); }
   ImproperVectorFEMassIntegrator(Coefficient &q) { Init(&q); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

void ImproperVectorFEMassIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
    MFEM_ASSERT(el.GetRangeType() == FiniteElement::SCALAR,
                "The improper vector FE should have a scalar type in the current implementation \n");

    int dim  = el.GetDim();
    int nd = el.GetDof();
    int improper_nd = nd * dim;

    double w;

#ifdef MFEM_THREAD_SAFE
    Vector scalar_shape.SetSize(nd);
    DenseMatrix vector_vshape(improper_nd, dim);
#else
    scalar_shape.SetSize(nd);
    vector_vshape.SetSize(improper_nd, dim);
#endif
    elmat.SetSize (improper_nd, improper_nd);

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

       Trans.SetIntPoint (&ip);

       el.CalcShape(ip, scalar_shape);
       for (int d = 0; d < dim; ++d )
           for (int l = 0; l < dim; ++l)
               for (int k = 0; k < nd; ++k)
               {
                   if (l == d)
                       vector_vshape(l*nd+k,d) = scalar_shape(k);
                   else
                       vector_vshape(l*nd+k,d) = 0.0;
               }
       // now vector_vshape is of size improper_nd x dim
       //el.CalcVShape(Trans, vector_vshape); // would be easy but this doesn't work for improper vector L2

       w = ip.weight * Trans.Weight();
       if (Q)
          w *= Q->Eval(Trans, ip);

       AddMult_a_AAt (w, vector_vshape, elmat);

    }
}

/// Integrator for (q * u, v)
/// where q is a scalar coefficient, u and v are from vector FE space created
/// from scalar FE collection (called improper vector FE)
class MixedImproperVectorMassIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient *Q;
   void Init(Coefficient *q)
   { Q = q; }

#ifndef MFEM_THREAD_SAFE
   Vector scalar_testshape;
   DenseMatrix test_vshape;
   Vector scalar_trialshape;
   DenseMatrix trial_vshape; // components are test shapes
#endif

public:
   MixedImproperVectorMassIntegrator() { Init(NULL); }
   MixedImproperVectorMassIntegrator(Coefficient *_q) { Init(_q); }
   MixedImproperVectorMassIntegrator(Coefficient &q) { Init(&q); }

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

void MixedImproperVectorMassIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
    // checking that both vector f.e. are improper vector f.e., i.e., have a scalar type in MFEM
    MFEM_ASSERT(test_fe.GetRangeType() == FiniteElement::SCALAR && trial_fe.GetRangeType() == FiniteElement::SCALAR,
                "Both vector f.e. should be improper vector f.e., i.e. have a scalar type in the current implementation \n");

    int dim  = test_fe.GetDim();
    int trial_dof = trial_fe.GetDof();
    int improper_trialdof = dim * trial_dof;
    int test_dof = test_fe.GetDof();
    int improper_testdof = dim * test_dof;

    double w;

#ifdef MFEM_THREAD_SAFE
    Vector scalar_testshape(test_dof);
    Vector scalar_trialshape(trial_dof);
    DenseMatrix trial_vshape(improper_trialdof,dim);
    DenseMatrix test_vshape(improper_testdof,dim);
#else
    trial_vshape.SetSize(improper_trialdof,dim);
    test_vshape.SetSize(improper_testdof,dim);
    scalar_testshape.SetSize(test_dof);
    scalar_trialshape.SetSize(test_dof);
#endif

    elmat.SetSize (improper_testdof, improper_trialdof);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
       int order = (Trans.OrderW() + test_fe.GetOrder() + trial_fe.GetOrder());
       ir = &IntRules.Get(test_fe.GetGeomType(), order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir->IntPoint(i);

       Trans.SetIntPoint (&ip);

       //trial_fe.CalcVShape(Trans, trial_vshape); // would be easy but this doesn't work for improper vector L2
       // so scalar shape functions are used here
       trial_fe.CalcShape(ip, scalar_trialshape);
       for (int d = 0; d < dim; ++d )
           for (int l = 0; l < dim; ++l)
               for (int k = 0; k < trial_dof; ++k)
               {
                   if (l == d)
                       trial_vshape(l*trial_dof+k,d) = scalar_trialshape(k);
                   else
                       trial_vshape(l*trial_dof+k,d) = 0.0;
               }

       //test_fe.CalcVShape(Trans, test_vshape); // would be easy but this doesn't work for improper vector L2
       // so scalar shape functions are used here
       test_fe.CalcShape(ip, scalar_testshape);
       for (int d = 0; d < dim; ++d )
           for (int l = 0; l < dim; ++l)
               for (int k = 0; k < test_dof; ++k)
               {
                   if (l == d)
                       test_vshape(l*test_dof+k,d) = scalar_testshape(k);
                   else
                       test_vshape(l*test_dof+k,d) = 0.0;
               }
       // now test_vshape is of size trial_dof(scalar)*dim x dim = improper_trialdof x dim

       w = ip.weight * Trans.Weight();
       if (Q)
          w *= Q->Eval(Trans, ip);

       AddMult_a_ABt (w, test_vshape, trial_vshape, elmat);

    }

}


/// Integrator for (q * u, v)
/// where q is a scalar coefficient, u is from vector FE space created
/// from scalar FE collection (called improper vector FE) and v is from
/// proper vector FE space (like RT or ND)
class MixedVectorVectorFEMassIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient *Q;
   void Init(Coefficient *q)
   { Q = q; }

#ifndef MFEM_THREAD_SAFE
   DenseMatrix test_vshape;
   Vector scalar_shape;
   DenseMatrix trial_vshape; // components are test shapes
#endif

public:
   MixedVectorVectorFEMassIntegrator() { Init(NULL); }
   MixedVectorVectorFEMassIntegrator(Coefficient *_q) { Init(_q); }
   MixedVectorVectorFEMassIntegrator(Coefficient &q) { Init(&q); }

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

void MixedVectorVectorFEMassIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
    // here we assume for a moment that proper vector FE is the trial one
    MFEM_ASSERT(test_fe.GetRangeType() == FiniteElement::SCALAR,
                "The improper vector FE should have a scalar type in the current implementation \n");

    int dim  = test_fe.GetDim();
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();
    int improper_testdof = dim * test_dof;

    double w;

#ifdef MFEM_THREAD_SAFE
    DenseMatrix trial_vshape(trial_dof, dim);
    DenseMatrix test_vshape(improper_testdof,dim);
    Vector scalar_shape(test_dof);
#else
    trial_vshape.SetSize(trial_dof, dim);
    test_vshape.SetSize(improper_testdof,dim);
    scalar_shape.SetSize(test_dof);
#endif

    elmat.SetSize (improper_testdof, trial_dof);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
       int order = (Trans.OrderW() + test_fe.GetOrder() + trial_fe.GetOrder());
       ir = &IntRules.Get(test_fe.GetGeomType(), order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir->IntPoint(i);

       Trans.SetIntPoint (&ip);

       trial_fe.CalcVShape(Trans, trial_vshape);

       test_fe.CalcShape(ip, scalar_shape);
       for (int d = 0; d < dim; ++d )
           for (int l = 0; l < dim; ++l)
               for (int k = 0; k < test_dof; ++k)
               {
                   if (l == d)
                       test_vshape(l*test_dof+k,d) = scalar_shape(k);
                   else
                       test_vshape(l*test_dof+k,d) = 0.0;
               }
       // now test_vshape is of size trial_dof(scalar)*dim x dim
       //test_fe.CalcVShape(Trans, test_vshape); // would be easy but this doesn't work for improper vector L2

       //std::cout << "trial_vshape \n";
       //trial_vshape.Print();

       //std::cout << "scalar_shape \n";
       //scalar_shape.Print();
       //std::cout << "improper test_vshape \n";
       //test_vshape.Print();

       w = ip.weight * Trans.Weight();
       if (Q)
          w *= Q->Eval(Trans, ip);

       for (int l = 0; l < dim; ++l)
       {
          for (int j = 0; j < test_dof; j++)
          {
             for (int k = 0; k < trial_dof; k++)
             {
                 for (int d = 0; d < dim; d++)
                 {
                    elmat(l*test_dof+j, k) += w * test_vshape(l*test_dof+j, d) * trial_vshape(k, d);
                 }
             }
          }
       }

       //std::cout << "elmat \n";
       //elmat.Print();
       //int p = 2;
       //p++;

    }

}

/// Integrator for (Q u, v)
/// where Q is a vector coefficient, u is from vector FE space created
/// from scalar FE collection and v is from scalar FE space
class MixedVectorScalarIntegrator : public BilinearFormIntegrator
{
private:
   VectorCoefficient *VQ;
   void Init(VectorCoefficient *vq)
   { VQ = vq; }

#ifndef MFEM_THREAD_SAFE
   Vector shape;
   Vector D;
   Vector test_shape;
   Vector b;
   Vector trial_shape;
   DenseMatrix trial_vshape; // components are test shapes
#endif

public:
   MixedVectorScalarIntegrator() { Init(NULL); }
   MixedVectorScalarIntegrator(VectorCoefficient *_vq) { Init(_vq); }
   MixedVectorScalarIntegrator(VectorCoefficient &vq) { Init(&vq); }

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

void MixedVectorScalarIntegrator::AssembleElementMatrix2(
        const FiniteElement &trial_fe, const FiniteElement &test_fe,
        ElementTransformation &Trans, DenseMatrix &elmat)
{
    // assume trial_fe is vector FE but created from scalar f.e. collection,
    // and test_fe is scalar FE

    MFEM_ASSERT(test_fe.GetRangeType() == FiniteElement::SCALAR && trial_fe.GetRangeType() == FiniteElement::SCALAR,
                "The improper vector FE should have a scalar type in the current implementation \n");

    int dim  = test_fe.GetDim();
    //int vdim = dim;
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();
    double w;

    if (VQ == NULL)
        mfem_error("MixedVectorScalarIntegrator::AssembleElementMatrix2(...)\n"
                "   is not implemented for non-vector coefficients");

#ifdef MFEM_THREAD_SAFE
    Vector trial_shape(trial_dof);
    DenseMatrix test_vshape(test_dof,dim);
#else
    trial_vshape.SetSize(trial_dof*dim,dim);
    test_shape.SetSize(test_dof);
#endif
    elmat.SetSize (test_dof, trial_dof * dim);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        int order = (Trans.OrderW() + test_fe.GetOrder() + trial_fe.GetOrder());
        ir = &IntRules.Get(test_fe.GetGeomType(), order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        test_fe.CalcShape(ip, test_shape);

        Trans.SetIntPoint (&ip);
        trial_fe.CalcShape(ip, trial_shape);

        //std::cout << "trial_shape \n";
        //trial_shape.Print();

        for (int d = 0; d < dim; ++d )
            for (int l = 0; l < dim; ++l)
                for (int k = 0; k < trial_dof; ++k)
                {
                    if (l == d)
                        trial_vshape(l*trial_dof+k,d) = trial_shape(k);
                    else
                        trial_vshape(l*trial_dof+k,d) = 0.0;
                }
        // now trial_vshape is of size trial_dof(scalar)*dim x dim

        //trial_fe.CalcVShape(Trans, trial_vshape); would be nice if it worked but no

        w = ip.weight * Trans.Weight();
        VQ->Eval (b, Trans, ip);

        for (int l = 0; l < dim; ++l)
            for (int j = 0; j < trial_dof; j++)
                for (int k = 0; k < test_dof; k++)
                    for (int d = 0; d < dim; d++ )
                    {
                        elmat(k, l*trial_dof + j) += w*trial_vshape(l*trial_dof + j,d)*b(d)*test_shape(k);
                    }
    }
}

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
      //double val = Tr.Weight() * Q.Eval(Tr, ip);
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

   dshape.SetSize(dof,dim);
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
      int order = (Tr.OrderW() + el.GetOrder() + el.GetOrder());
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Tr.SetIntPoint (&ip);
      w = ip.weight;// * Tr.Weight();
      CalcAdjugate(Tr.Jacobian(), invdfdx);
      Mult(dshape, invdfdx, dshapedxt);

      Q.Eval(bf, Tr, ip);

      dshapedxt.Mult(bf, bfdshapedxt);

      add(elvect, w, bfdshapedxt, elvect);
   }
}

// Some functions used for analytical tests previously

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

// Templates for Transport_test class

template <double (*S)(const Vector&), void (*bvec)(const Vector&, Vector& )>
    void sigmaTemplate(const Vector& xt, Vector& sigma);

template <void (*bvec)(const Vector&, Vector& )>
    void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda);

template <void (*bvec)(const Vector&, Vector& )>
        void bbTTemplate(const Vector& xt, DenseMatrix& bbT);

template <void (*bvec)(const Vector&, Vector& )>
    double bTbTemplate(const Vector& xt);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), \
         void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void bfTemplate(const Vector& xt, Vector& bf);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), \
         void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        double divsigmaTemplate(const Vector& xt);

class Transport_test
{
    protected:
        int dim;
        int numsol;

    public:
        FunctionCoefficient * scalarS;
        FunctionCoefficient * scalardivsigma;         // = dS/dt + div (bS) = div sigma
        FunctionCoefficient * bTb;
        VectorFunctionCoefficient * sigma;
        VectorFunctionCoefficient * b;
        VectorFunctionCoefficient * bf;
        MatrixFunctionCoefficient * Ktilda;
        MatrixFunctionCoefficient * bbT;
    public:
        Transport_test (int Dim, int NumSol);

        int GetDim() {return dim;}
        int GetNumSol() {return numsol;}
        void SetDim(int Dim) { dim = Dim;}
        void SetNumSol(int NumSol) { numsol = NumSol;}
        bool CheckTestConfig();

        ~Transport_test () {}
    private:
        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), \
                 void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbvec)(const Vector & xt)> \
        void SetTestCoeffs ( );

        void SetScalarSFun( double (*S)(const Vector & xt))
        { scalarS = new FunctionCoefficient(S);}

        template< void(*bvec)(const Vector & x, Vector & vec)>  \
        void SetScalarBtBFun()
        {
            bTb = new FunctionCoefficient(bTbTemplate<bvec>);
        }

        template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec)> \
        void SetSigmaVec()
        {
            sigma = new VectorFunctionCoefficient(dim, sigmaTemplate<S,bvec>);
        }

        void SetbVec( void(*bvec)(const Vector & x, Vector & vec))
        { b = new VectorFunctionCoefficient(dim, bvec);}

        template< void(*bvec)(const Vector & x, Vector & vec)>  \
        void SetKtildaMat()
        {
            Ktilda = new MatrixFunctionCoefficient(dim, KtildaTemplate<bvec>);
        }

        template< void(*bvec)(const Vector & x, Vector & vec)>  \
        void SetBBtMat()
        {
            bbT = new MatrixFunctionCoefficient(dim, bbTTemplate<bvec>);
        }

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), \
                 void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        void SetbfVec()
        { bf = new VectorFunctionCoefficient(dim, bfTemplate<S,dSdt,Sgradxvec,bvec,divbfunc>);}

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), \
                 void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void SetDivSigmaFun()
        { scalardivsigma = new FunctionCoefficient(divsigmaTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>);}

};

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), \
         void(*Sgradxvec)(const Vector & x, Vector & gradx),  \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
void Transport_test::SetTestCoeffs ()
{
    SetScalarSFun(S);
    SetbVec(bvec);
    SetbfVec<S, dSdt, Sgradxvec, bvec, divbfunc>();
    SetSigmaVec<S,bvec>();
    SetKtildaMat<bvec>();
    SetScalarBtBFun<bvec>();
    SetDivSigmaFun<S, dSdt, Sgradxvec, bvec, divbfunc>();
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
    int par_ref_levels  = 1;

    const char *formulation = "cfosls"; // "cfosls" or "fosls"
    const char *sigmaspace = "H1"; // or vector "H1"

    // solver options
    int prec_option = 0; //defines whether to use preconditioner or not, and which one. (*) 4 and 5 produce almost the same results
    bool direct_solver = false;
    bool with_prec;           // to be defined from prec_option value
    bool ADS_for_A11;         // whether to use ADS for A11 block
    bool identity_Schur11;    // if true, uses identity for Schur complement
    bool identity_Schur22;    // if true, uses identity for Schur complement

    // level gap between coarse an fine grid (T_H and T_h)
    int level_gap = 1;

    const char *mesh_file = "../data/cube_3d_moderate.mesh";

    //const char * mesh_file = "../data/cube4d_low.MFEM";
    //const char * mesh_file = "../data/cube4d.MFEM";
    //const char * mesh_file = "dsadsad";
    //const char * mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";
    //const char * mesh_file = "../data/square_2d_moderate.mesh";

    //const char * meshbase_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * meshbase_file = "../data/sphere3D_0.05to0.1.mesh";
    //const char * meshbase_file = "../data/sphere3D_veryfine.mesh";
    //const char * meshbase_file = "../data/beam-tet.mesh";
    //const char * meshbase_file = "../data/orthotope3D_moderate.mesh";
    //const char * meshbase_file = "../data/orthotope3D_fine.mesh";
    //const char * meshbase_file = "../data/square_2d_moderate.mesh";
    //const char * meshbase_file = "../data/square_2d_fine.mesh";
    //const char * meshbase_file = "dsadsad";
    //const char * meshbase_file = "../data/circle_fine_0.1.mfem";
    //const char * meshbase_file = "../data/circle_moderate_0.2.mfem";

    int feorder         = 0;

    if (verbose)
        cout << "Solving within СFOSLS & Petrov-Galerkin formulation of the transport equation with MFEM \n";

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&feorder, "-o", "--feorder",
                   "Finite element order (polynomial degree).");
    args.AddOption(&ser_ref_levels, "-sref", "--sref",
                   "Number of serial refinements 4d mesh.");
    args.AddOption(&par_ref_levels, "-pref", "--pref",
                   "Number of parallel refinements 4d mesh.");
    args.AddOption(&nDimensions, "-dim", "--whichD",
                   "Dimension of the space-time problem.");
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
    args.AddOption(&level_gap, "-lg", "--level-gap",
                   "level gap between coarse an fine grid (T_H and T_h).");

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

    switch (prec_option)
    {
    case 1:
        with_prec = true;
        ADS_for_A11 = false;
        identity_Schur11 = false;
        identity_Schur22 = false;
        break;
    case 2: // ADS for H(div)
        with_prec = true;
        ADS_for_A11 = true;
        identity_Schur11 = false;
        identity_Schur22 = false;
        break;
    case 3:
        with_prec = true;
        ADS_for_A11 = false;
        identity_Schur11 = true;
        identity_Schur22 = false;
        break;
    case 4:
        with_prec = true;
        ADS_for_A11 = false;
        identity_Schur11 = false;
        identity_Schur22 = true;
        break;
    case 5:
        with_prec = true;
        ADS_for_A11 = false;
        identity_Schur11 = true;
        identity_Schur22 = true;
        break;
    default: // no preconditioner (default)
        with_prec = false;
        ADS_for_A11 = false;
        identity_Schur11 = false;
        break;
    }

    MFEM_ASSERT(strcmp(sigmaspace,"Hdiv")==0 || strcmp(sigmaspace,"H1")==0, "sigmaspace must be Hdiv or H1! \n");
    MFEM_ASSERT(!(ADS_for_A11 && strcmp(sigmaspace,"H1")==0), "Cannot use ADS for vector H1 (sigma space)! \n");


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
        if (verbose)
            cout << "Reading a " << nDimensions << "d mesh from the file " << mesh_file << "\n";
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
    else //if nDimensions is no 3 or 4
    {
        if (verbose)
            cerr << "Case nDimensions = " << nDimensions << " is not supported \n"
                 << flush;
        MPI_Finalize();
        return -1;

    }
    //mesh = new Mesh(2, 2, 2, Element::HEXAHEDRON, 1);

    if (mesh) // if only serial mesh was generated previously, parallel mesh is initialized here
    {
        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        if ( verbose )
            cout << "Creating parmesh(" << nDimensions <<
                    "d) from the serial mesh (" << nDimensions << "d) \n" << flush;
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }

    MFEM_ASSERT(level_gap>0 && level_gap<=par_ref_levels, "invalid level_gap!");
    for (int l = 0; l < par_ref_levels-level_gap; l++)
    {
       pmesh->UniformRefinement();
    }

    int dim = nDimensions;
    FiniteElementCollection *l2_coll = new L2_FECollection(feorder, dim);
    if(verbose)
        cout << "L2: order " << feorder << endl;

    ParFiniteElementSpace *L_space = new ParFiniteElementSpace(pmesh.get(), l2_coll, dim, Ordering::byVDIM); // space for lambda
    ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll); // space for mu
    ParFiniteElementSpace *coarseL_space = new ParFiniteElementSpace(pmesh.get(), l2_coll, dim, Ordering::byVDIM); // space for lambda
    ParFiniteElementSpace *coarseW_space = new ParFiniteElementSpace(pmesh.get(), l2_coll); // space for mu

    ParFiniteElementSpace *coarseL_space_help = new ParFiniteElementSpace(pmesh.get(), l2_coll, dim, Ordering::byVDIM); // space for lambda
    ParFiniteElementSpace *coarseW_space_help = new ParFiniteElementSpace(pmesh.get(), l2_coll); // space for mu

    HypreParMatrix * P_L, *P_W;
    Array<HypreParMatrix*> P_Ls(level_gap), P_Ws(level_gap);
    for (int l = 0; l < level_gap; l++)
    {
        coarseL_space_help->Update();
        coarseW_space_help->Update();

        pmesh->UniformRefinement();

        L_space->Update();
        W_space->Update();

        {
            auto d_td_coarse_L = coarseL_space_help->Dof_TrueDof_Matrix();
            auto P_L_loc_tmp = (SparseMatrix *)L_space->GetUpdateOperator();
            auto P_L_local = RemoveZeroEntries(*P_L_loc_tmp);
            unique_ptr<SparseMatrix>RP_L_local(
                        Mult(*L_space->GetRestrictionMatrix(), *P_L_local));

            if (level_gap==1)
            {
                P_L = d_td_coarse_L->LeftDiagMult(
                            *RP_L_local, L_space->GetTrueDofOffsets());
                P_L->CopyColStarts();
                P_L->CopyRowStarts();
            }
            else
            {
                P_Ls[l] = d_td_coarse_L->LeftDiagMult(
                            *RP_L_local, L_space->GetTrueDofOffsets());
                P_Ls[l]->CopyColStarts();
                P_Ls[l]->CopyRowStarts();
            }
            delete P_L_local;
        }

        {
            auto d_td_coarse_W = coarseW_space_help->Dof_TrueDof_Matrix();
            auto P_W_loc_tmp = (SparseMatrix *)W_space->GetUpdateOperator();
            auto P_W_local = RemoveZeroEntries(*P_W_loc_tmp);
            unique_ptr<SparseMatrix>RP_W_local(
                        Mult(*W_space->GetRestrictionMatrix(), *P_W_local));

            if (level_gap==1)
            {
                P_W = d_td_coarse_W->LeftDiagMult(
                            *RP_W_local, W_space->GetTrueDofOffsets());
                P_W->CopyColStarts();
                P_W->CopyRowStarts();
            }
            else
            {
                P_Ws[l] = d_td_coarse_W->LeftDiagMult(
                            *RP_W_local, W_space->GetTrueDofOffsets());
                P_Ws[l]->CopyColStarts();
                P_Ws[l]->CopyRowStarts();
            }
            delete P_W_local;
        }
    } // end of loop over mesh levels

    // Combine the interpolation matrices from different levels
    Array<HypreParMatrix*> help_Ls(level_gap-2), help_Ws(level_gap-2);
    if (level_gap > 2)
    {
        help_Ls[0] = ParMult(P_Ls[1],P_Ls[0]);
        help_Ws[0] = ParMult(P_Ws[1],P_Ws[0]);
    }
    else if (level_gap == 2)
    {
        P_L = ParMult(P_Ls[1],P_Ls[0]);
        P_W = ParMult(P_Ws[1],P_Ws[0]);
    }

    for (int l = 0; l < level_gap-3; l++)
    {
        help_Ls[l+1] = ParMult(P_Ls[l+2],help_Ls[l]);
        help_Ws[l+1] = ParMult(P_Ws[l+2],help_Ws[l]);
    }
    if (level_gap > 2)
    {
        P_L = ParMult(P_Ls[level_gap-1],help_Ls[level_gap-3]);
        P_W = ParMult(P_Ws[level_gap-1],help_Ws[level_gap-3]);
    }



    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.


    FiniteElementCollection *sigma_coll;
    if (strcmp(sigmaspace,"Hdiv") == 0)
    {
        if ( dim == 4 )
        {
            sigma_coll = new RT0_4DFECollection;
            if(verbose)
                cout << "RT: order 0 for 4D" << endl;
        }
        else
        {
            sigma_coll = new RT_FECollection(feorder, dim);
            if(verbose)
                cout << "RT: order " << feorder << " for 3D \n";
        }
    }
    else // vector H1
    {
        if (dim == 4)
        {
            sigma_coll = new LinearFECollection;
            if (verbose)
                cout << "H1 for sigma in 4D: linear elements are used \n";
        }
        else // dim == 3
        {
            sigma_coll = new H1_FECollection(feorder+1, dim);
            if(verbose)
                cout << "H1 for sigma: order " << feorder + 1 << " for 3D \n";
        }

    }

    if (dim == 4)
        MFEM_ASSERT(feorder==0, "Only lowest order elements are support in 4D!");
    FiniteElementCollection *h1_coll;
    if (dim == 4)
    {
        h1_coll = new LinearFECollection;
        if (verbose)
            cout << "H1 in 4D: linear elements are used \n";
    }
    else
    {
        h1_coll = new H1_FECollection(feorder+1, dim);
        if(verbose)
            cout << "H1: order " << feorder + 1 << " for 3D \n";
    }

    ParFiniteElementSpace *R_space;
    if (strcmp(sigmaspace,"Hdiv") == 0)
        R_space = new ParFiniteElementSpace(pmesh.get(), sigma_coll);
    else // H1
        R_space = new ParFiniteElementSpace(pmesh.get(), sigma_coll, dim, Ordering::byVDIM);
    ParFiniteElementSpace *H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);
    //ParFiniteElementSpace *L_space = new ParFiniteElementSpace(pmesh.get(), l2_coll, dim); // space for lambda
//    ParFiniteElementSpace *L_space = new ParFiniteElementSpace(pmesh.get(), l2_coll, dim, Ordering::byVDIM); // space for lambda
//    ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll); // space for mu

    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimH = H_space->GlobalTrueVSize();
    HYPRE_Int dimL = L_space->GlobalTrueVSize();
    HYPRE_Int dimW = W_space->GlobalTrueVSize();

    if (verbose)
    {
       std::cout << "***********************************************************\n";
       std::cout << "dim(R) = " << dimR << ", ";
       std::cout << "dim(H) = " << dimH << ", ";
       std::cout << "dim(L) = " << dimL << ", ";
       std::cout << "dim(W) = " << dimW << ", ";
       std::cout << "dim(R+H+W+L) = " << dimR + dimH + dimL + dimW << "\n";
       std::cout << "Number of primary unknowns (sigma and S): " << dimR + dimH << "\n";
       std::cout << "Number of equations in constraints: " << P_L->Width() + P_W->Width() << "\n";
       std::cout << "***********************************************************\n";
    }

    // 7. Define the two BlockStructure of the problem.  block_offsets is used
    //    for Vector based on dof (like ParGridFunction or ParLinearForm),
    //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
    //    for the rhs and solution of the linear system).  The offsets computed
    //    here are local to the processor.
    int numblocks = 4;

    Array<int> block_offsets(numblocks + 1); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = R_space->GetVSize();
    block_offsets[2] = H_space->GetVSize();
    block_offsets[3] = L_space->GetVSize();
    block_offsets[4] = W_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(numblocks + 1); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = R_space->TrueVSize();
    block_trueOffsets[2] = H_space->TrueVSize();
    block_trueOffsets[3] = L_space->TrueVSize();
    block_trueOffsets[4] = W_space->TrueVSize();
    block_trueOffsets.PartialSum();

    Array<int> block_finalOffsets(numblocks + 1); // number of variables + 1
    block_finalOffsets[0] = 0;
    block_finalOffsets[1] = R_space->TrueVSize();
    block_finalOffsets[2] = H_space->TrueVSize();
    block_finalOffsets[3] = coarseL_space->TrueVSize();
    block_finalOffsets[4] = coarseW_space->TrueVSize();
    block_finalOffsets.PartialSum();

    BlockVector x(block_offsets), rhs(block_offsets);
    BlockVector trueX(block_finalOffsets), trueRhs(block_trueOffsets);
    x = 0.0;
    rhs = 0.0;
    trueX = 0.0;
    trueRhs = 0.0;

    Transport_test Mytest(nDimensions,numsol);

    ParGridFunction *S_exact = new ParGridFunction(H_space);
    S_exact->ProjectCoefficient(*(Mytest.scalarS));

    ParGridFunction * sigma_exact = new ParGridFunction(R_space);
    sigma_exact->ProjectCoefficient(*(Mytest.sigma));

    x.GetBlock(0) = *sigma_exact;
    x.GetBlock(1) = *S_exact;

   // 8. Define the constant/function coefficients.
   ConstantCoefficient zero(.0);
   Vector zerovec(dim);
   zerovec = 0.0;
   VectorConstantCoefficient zerov(zerovec);

   //----------------------------------------------------------
   // Setting boundary conditions.
   //----------------------------------------------------------

   // for S boundary conditions are essential only for t = 0 in parabolic case
   Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
   ess_bdrS = 0;
   ess_bdrS[0] = 1; // t = 0
   Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());
   ess_bdrSigma = 0;
   //ess_bdr[1] = 1; // lateral boundary
   //-----------------------

   // 9. Define the parallel grid function and parallel linear forms, solution
   //    vector and rhs.

   ParLinearForm *rhssig_form(new ParLinearForm);
   rhssig_form->Update(R_space, rhs.GetBlock(0), 0);
   if (strcmp(sigmaspace,"Hdiv") == 0)
       rhssig_form->AddDomainIntegrator(new VectorFEDomainLFIntegrator(zerov));
   else
       rhssig_form->AddDomainIntegrator(new VectorDomainLFIntegrator(zerov));
   rhssig_form->Assemble();
   //rhssig_form->Print(std::cout);

   ParLinearForm *rhsS_form(new ParLinearForm);
   rhsS_form->Update(H_space, rhs.GetBlock(1), 0);
   rhsS_form->AddDomainIntegrator(new DomainLFIntegrator(zero));
   rhsS_form->Assemble();
   //rhsS_form->Print(std::cout);

   ParLinearForm *rhslam_form(new ParLinearForm);
   rhslam_form->Update(L_space, rhs.GetBlock(2), 0);
   rhslam_form->AddDomainIntegrator(new VectorDomainLFIntegrator(zerov));
   rhslam_form->Assemble();
   //rhslam_form->Print(std::cout);

   ParLinearForm *rhsmu_form(new ParLinearForm);
   rhsmu_form->Update(W_space, rhs.GetBlock(3), 0);
   rhsmu_form->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
   rhsmu_form->Assemble();

   // 10. Assemble the finite element matrices for the CFOSLS operator

   //---------------
   //  A Block: block-diagonal with stiffness matrices for H(div) and H1
   //---------------

   //----------------
   //  A11 Block: (sigma, phi) + (div sigma, div phi) with sigma and phi from H(div)
   //-----------------

   ParBilinearForm *A11_block(new ParBilinearForm(R_space));
   HypreParMatrix *A11;
   if (strcmp(sigmaspace,"Hdiv") == 0)
   {
       A11_block->AddDomainIntegrator(new VectorFEMassIntegrator);
       A11_block->AddDomainIntegrator(new DivDivIntegrator);
   }
   else
   {
       A11_block->AddDomainIntegrator(new VectorMassIntegrator);
       A11_block->AddDomainIntegrator(new ImproperDivDivIntegrator);
   }
   A11_block->Assemble();
   A11_block->EliminateEssentialBC(ess_bdrSigma, x.GetBlock(0), *rhssig_form);
   A11_block->Finalize();
   A11 = A11_block->ParallelAssemble();

   //----------------
   //  A22 Block: (S, p) + (grad S, grad p) with S and p from H1
   //-----------------

   ParBilinearForm *A22_block(new ParBilinearForm(H_space));
   HypreParMatrix *A22;
   A22_block->AddDomainIntegrator(new MassIntegrator);
   A22_block->AddDomainIntegrator(new DiffusionIntegrator);
   A22_block->Assemble();
   A22_block->EliminateEssentialBC(ess_bdrS, x.GetBlock(1), *rhsS_form);
   A22_block->Finalize();
   A22 = A22_block->ParallelAssemble();

   //---------------
   //  B Block: block 2 x 2 with Lagrange multiplier's stuff
   //---------------

   //----------------
   //  B11 Block: (sigma, theta) with sigma from H(div) and theta from vector L2
   //-----------------

   ParMixedBilinearForm *B11block(new ParMixedBilinearForm(R_space, L_space));
   HypreParMatrix *B11, *B11T;
   if (strcmp(sigmaspace,"Hdiv") == 0)
       B11block->AddDomainIntegrator(new MixedVectorVectorFEMassIntegrator); // FIXME: Standard VectorFEMassIntegrator shouldn't work incorrectly
   else // H1
   {
       B11block->AddDomainIntegrator(new MixedImproperVectorMassIntegrator);
   }
   //B11block->AddDomainIntegrator(new VectorFEMassIntegrator);    // but this gives the same result, why?
   B11block->Assemble();
   B11block->EliminateTrialDofs(ess_bdrSigma, x.GetBlock(0), *rhslam_form);
   B11block->Finalize();
   auto B11tmp = B11block->ParallelAssemble();
   B11 = ParMult(P_L->Transpose(), B11tmp);
   B11T = B11->Transpose();

   //----------------
   //  B12 Block: -([b,1]^T S, theta) with S is from H1 and theta from vector L2
   //-----------------

   ParMixedBilinearForm *B12Tblock(new ParMixedBilinearForm(L_space, H_space));
   HypreParMatrix *B12, *B12T;
   B12Tblock->AddDomainIntegrator(new MixedVectorScalarIntegrator(*Mytest.b)); // FIXME: are Jacobians missing?
   //B12Tblock->AddDomainIntegrator(new MixedScalarVectorIntegrator(*Mytest.b));
   //B12Tblock->AddDomainIntegrator(new MixedDotProductIntegrator(*Mytest.b));
   B12Tblock->Assemble();
   B12Tblock->EliminateTestDofs(ess_bdrS);
   B12Tblock->Finalize();
   auto B12Ttmp = B12Tblock->ParallelAssemble();
   *B12Ttmp *= -1.;
   B12T = ParMult(B12Ttmp, P_L);

   B12 = B12T->Transpose();

   //----------------
   //  B21 Block: divergence constraint
   //-----------------

   HypreParMatrix *B21;
   HypreParMatrix *B21T;

   ParMixedBilinearForm *B21block(new ParMixedBilinearForm(R_space, W_space));
   if (strcmp(sigmaspace,"Hdiv") == 0)
       B21block->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   else // H1
       // looks like it already existed in MFEM, for impropver vector FE as we need here
       B21block->AddDomainIntegrator(new VectorDivergenceIntegrator);
   B21block->Assemble();
   B21block->EliminateTrialDofs(ess_bdrSigma, x.GetBlock(0), *rhsmu_form);
   B21block->Finalize();
   auto B21tmp = B21block->ParallelAssemble();
   B21 = ParMult(P_W->Transpose(), B21tmp);

   B21T = B21->Transpose();

#ifdef DEBUGGING
   ParBilinearForm *L33_block(new ParBilinearForm(L_space));
   HypreParMatrix *L33;
   L33_block->AddDomainIntegrator(new ImproperVectorFEMassIntegrator);
   L33_block->Assemble();
   L33_block->Finalize();
   L33 = L33_block->ParallelAssemble();

   ParBilinearForm *W44_block(new ParBilinearForm(W_space));
   HypreParMatrix *W44;
   W44_block->AddDomainIntegrator(new MassIntegrator);
   W44_block->Assemble();
   W44_block->Finalize();
   W44 = W44_block->ParallelAssemble();
#endif
   //=======================================================
   // Assembling the righthand side
   //-------------------------------------------------------

  rhssig_form->ParallelAssemble(trueRhs.GetBlock(0));
  rhsS_form->ParallelAssemble(trueRhs.GetBlock(1));
  rhslam_form->ParallelAssemble(trueRhs.GetBlock(2));
  rhsmu_form->ParallelAssemble(trueRhs.GetBlock(3));

  //========================================================
  // Checking residuals in the constraints on exact solutions (serial version)
  //--------------------------------------------------------

  Vector TrueSig(R_space->TrueVSize()), TrueS(H_space->TrueVSize());
  sigma_exact->ParallelProject(TrueSig);
  S_exact->ParallelProject(TrueS);

  Vector resL(coarseL_space->TrueVSize()), resW(coarseW_space->TrueVSize());
  Vector tempL1(coarseL_space->TrueVSize()), tempL2(coarseW_space->TrueVSize());
  B21->Mult(TrueSig, resW);
  P_W->MultTranspose(trueRhs.GetBlock(3), tempL2);
  //resW -= trueRhs.GetBlock(3);
  resW -= tempL2;

  B11->Mult(TrueSig,tempL1);
  std::cout << "|| B11 * sigma_exact || = " << tempL1.Norml2() / sqrt (tempL1.Size()) << "\n";
  B12->Mult(TrueS, resL);
  //resL.Print();
  std::cout << "|| B12 * S_exact || = " << resL.Norml2() / sqrt (resL.Size()) << "\n";
  resL += tempL1;

  double norm_resW = resW.Norml2() / sqrt (resW.Size());
  double norm_resL = resL.Norml2() / sqrt (resL.Size());
  double norm_rhsW = trueRhs.GetBlock(3).Norml2() / sqrt (trueRhs.GetBlock(3).Size());

  std::cout << "Residuals in constraints for exact solution: \n";
  std::cout << "norm_resL = " << norm_resL << "\n";
  std::cout << "norm_resW = " << norm_resW << "\n";
  std::cout << "rel. norm_resW = " << norm_resW / norm_rhsW << "\n";

  //========================================================
  // Checking functional on exact solutions
  //--------------------------------------------------------

  Vector energySig(R_space->TrueVSize()), energyS(H_space->TrueVSize());
  A11->Mult(TrueSig, energySig);
  A22->Mult(TrueS, energyS);

  double local_energySig = energySig * TrueSig;
  double local_energyS = energyS * TrueS;

  double global_energySig;
  MPI_Reduce(&local_energySig, &global_energySig, 1,
             MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  double global_energyS;
  MPI_Reduce(&local_energyS, &global_energyS, 1,
             MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (verbose)
  {
      std::cout << "TrueSig H(div) norm = " << global_energySig << "\n";
      std::cout << "TrueS H1 norm = " << global_energyS << "\n";
      std::cout << "Quadratic functional on exact solution = " << global_energySig * global_energySig
                  + global_energyS * global_energyS << "\n";
  }

  //MPI_Finalize();
  //return 0;

  //=======================================================
  // Assembling the Matrix
  //-------------------------------------------------------

  BlockOperator *CFOSLSop = new BlockOperator(block_finalOffsets);
  CFOSLSop->SetBlock(0,0, A11);
  CFOSLSop->SetBlock(1,1, A22);
  CFOSLSop->SetBlock(0,2, B11T);
  CFOSLSop->SetBlock(0,3, B21T);
  CFOSLSop->SetBlock(1,2, B12T);
  CFOSLSop->SetBlock(2,0, B11);
  CFOSLSop->SetBlock(2,1, B12);
  CFOSLSop->SetBlock(3,0, B21);

   if (verbose)
       cout << "Final saddle point matrix assembled \n" << flush;
   MPI_Barrier(MPI_COMM_WORLD);

   // 12. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.

   BlockDiagonalPreconditioner prec(block_finalOffsets);
   if (with_prec > 0 && !direct_solver)
   {
       // Construct the operators for preconditioner
       if (verbose)
           cout << "Using a block diagonal preconditioner \n";
       chrono.Clear();
       chrono.Start();

       Solver * invA11;
       if (ADS_for_A11)
       {
           invA11 = new HypreADS(*A11, R_space);
       }
       else // BoomerAMG
       {
           invA11 = new HypreBoomerAMG(*A11);
       }
       ((HypreBoomerAMG*)invA11)->SetPrintLevel(0);
       ((HypreBoomerAMG*)invA11)->iterative_mode = false;

       HypreBoomerAMG * invA22 = new HypreBoomerAMG(*A22);
       invA22->iterative_mode = false;
       invA22->SetPrintLevel(0);

       HypreParVector *A11d = new HypreParVector(MPI_COMM_WORLD, A11->GetGlobalNumRows(), A11->GetRowStarts());
       A11->GetDiag(*A11d);
       HypreParVector *A22d = new HypreParVector(MPI_COMM_WORLD, A22->GetGlobalNumRows(), A22->GetRowStarts());
       A22->GetDiag(*A22d);

       Operator * invLam;
       if (!identity_Schur11)
       {
           HypreParMatrix * Temp11 = B11->Transpose();
           Temp11->InvScaleRows(*A11d);
           HypreParMatrix * Tempp = ParMult(B11, Temp11);
           HypreParMatrix * Temp12 = B12->Transpose();
           Temp12->InvScaleRows(*A22d);
           HypreParMatrix * Schur11 = ParMult(B12, Temp12);
           //*Tempp += *Schur11; //->Add(1.0, *Schur11);
           *Schur11 += *Tempp; // cannot interchange matrices, gives internal hypre error, maybe because of sparsity patterns

           invLam = new HypreBoomerAMG(*Schur11);
           ((HypreBoomerAMG *)invLam)->SetPrintLevel(0);
           ((HypreBoomerAMG *)invLam)->iterative_mode = false;
       }
       else
       {
           invLam = new IdentityOperator(B12->Height());
       }

       Operator * invMu;
       if (!identity_Schur22)
       {
           HypreParMatrix * Temp22 = B21->Transpose();
           Temp22->InvScaleRows(*A11d);
           HypreParMatrix * Schur22 = ParMult(B21, Temp22);

           invMu = new HypreBoomerAMG(*Schur22);
           ((HypreBoomerAMG *)invMu)->SetPrintLevel(0);
           ((HypreBoomerAMG *)invMu)->iterative_mode = false;
       }
       else
       {
           invMu = new IdentityOperator(B21->Height());
       }

       prec.SetDiagonalBlock(0, invA11);
       prec.SetDiagonalBlock(1, invA22);
       prec.SetDiagonalBlock(2, invLam);
       prec.SetDiagonalBlock(3, invMu);

       if (verbose)
           std::cout << "Preconditioner built in " << chrono.RealTime() << "s. \n";
   }
   else
       if (verbose)
           cout << "No preconditioner is used. \n";

   // 12. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.

   chrono.Clear();
   chrono.Start();

   //GMRESSolver solver(MPI_COMM_WORLD);    // too slow
   MINRESSolver solver(MPI_COMM_WORLD);
#ifdef MFEM_USE_SUITESPARSE
   Solver * DirectSolver;
   BlockMatrix * SystemBlockMat;
   SparseMatrix * SystemMat;
   SparseMatrix *A11_diag, *A22_diag, *B11_diag, *B12_diag, *B21_diag, *B11T_diag, *B12T_diag, *B21T_diag;
#ifdef DEBUGGING
   SparseMatrix *L33_diag, *W44_diag;
#endif
#endif
   if (direct_solver)
   {
#ifdef MFEM_USE_SUPERLU
       if (verbose)
           std::cout << "Using parallel SuperLU direct solver \n";
       Operator * LURowOp = new SuperLURowLocMatrix(*CFOSLSop);

       SuperLUSolver * superlu = new SuperLUSolver(MPI_COMM_WORLD);
       superlu->SetPrintStatistics(false);
       superlu->SetSymmetricPattern(true);
       superlu->SetColumnPermutation(superlu::PARMETIS);
       superlu->SetOperator(*LURowOp);
       solver.SetPreconditioner(superlu);
#else
#ifdef MFEM_USE_SUITESPARSE
       if (verbose)
           std::cout << "Using serial UMFPack direct solver \n";
       SystemBlockMat = new BlockMatrix(block_finalOffsets);
       A11_diag = new SparseMatrix();
       A22_diag = new SparseMatrix();
       B11_diag = new SparseMatrix();
       B12_diag = new SparseMatrix();
       B21_diag = new SparseMatrix();
       B11T_diag = new SparseMatrix();
       B12T_diag = new SparseMatrix();
       B21T_diag = new SparseMatrix();
       A11->GetDiag(*A11_diag);
       A22->GetDiag(*A22_diag);
       B11->GetDiag(*B11_diag);
       B12->GetDiag(*B12_diag);
       B21->GetDiag(*B21_diag);
       B11T->GetDiag(*B11T_diag);
       B12T->GetDiag(*B12T_diag);
       B21T->GetDiag(*B21T_diag);

#ifdef DEBUGGING
       L33_diag = new SparseMatrix();
       L33->GetDiag(*L33_diag);
       W44_diag = new SparseMatrix();
       W44->GetDiag(*W44_diag);
#endif

       std::cout << "infinitness measure, A11 = " << A11_diag->CheckFinite() << "\n";
       std::cout << "infinitness measure, A22 = " << A22_diag->CheckFinite() << "\n";
       std::cout << "infinitness measure, B11 = " << B11_diag->CheckFinite() << "\n";
       std::cout << "infinitness measure, B12 = " << B12_diag->CheckFinite() << "\n";
       std::cout << "infinitness measure, B21 = " << B21_diag->CheckFinite() << "\n";
       std::cout << "infinitness measure, B11T = " << B11T_diag->CheckFinite() << "\n";
       std::cout << "infinitness measure, B12T = " << B12T_diag->CheckFinite() << "\n";
       std::cout << "infinitness measure, B21T = " << B21T_diag->CheckFinite() << "\n";
       std::cout << "norm, A11 = " << A11_diag->MaxNorm() << "\n";
       std::cout << "norm, A22 = " << A22_diag->MaxNorm() << "\n";
       std::cout << "norm, B11 = " << B11_diag->MaxNorm() << "\n";
       std::cout << "norm, B12 = " << B12_diag->MaxNorm() << "\n";
       std::cout << "norm, B21 = " << B21_diag->MaxNorm() << "\n";

       SystemBlockMat->SetBlock(0,0,A11_diag);
       SystemBlockMat->SetBlock(1,1,A22_diag);
#ifndef DEBUGGING
       SystemBlockMat->SetBlock(0,2,B11T_diag);
       SystemBlockMat->SetBlock(0,3,B21T_diag);
       SystemBlockMat->SetBlock(1,2,B12T_diag);
       SystemBlockMat->SetBlock(2,0,B11_diag);
       SystemBlockMat->SetBlock(2,1,B12_diag);
       SystemBlockMat->SetBlock(3,0,B21_diag);
#else
       SystemBlockMat->SetBlock(2,2,L33_diag);
       SystemBlockMat->SetBlock(3,3,W44_diag);
#endif
       SystemMat = SystemBlockMat->CreateMonolithic();

       DirectSolver = new UMFPackSolver(*SystemMat);
       DirectSolver->iterative_mode = false;
       ((UMFPackSolver*)DirectSolver)->SetPrintLevel(10);

       std::cout << "unsymmetry measure = " << SystemMat->IsSymmetric() << "\n";
       std::cout << "infinitness measure = " << SystemMat->CheckFinite() << "\n";
       //SystemMat->Print();
       solver.SetOperator(*SystemMat);
//       solver.SetPreconditioner(*DirectSolver);
#else
       if (verbose)
            std::cout << "Error: no suitesparse and no superlu, direct solver cannot be used \n";
       MPI_Finalize();
       return 0;
#endif
#endif
   } // end of case when direct solver is used
   else // iterative solver
   {
       solver.SetAbsTol(atol);
       solver.SetRelTol(rtol);
       solver.SetMaxIter(max_iter);
       solver.SetOperator(*CFOSLSop); // overwritten in case of UMFPackSolver
       if (with_prec > 0 && !direct_solver )
            solver.SetPreconditioner(prec);
   }
   trueX = 0.0;
   BlockVector finalRhs(block_finalOffsets);
   finalRhs = 0.0;
   finalRhs.GetBlock(0) = trueRhs.GetBlock(0);
   finalRhs.GetBlock(1) = trueRhs.GetBlock(1);
   P_L->MultTranspose(trueRhs.GetBlock(2), finalRhs.GetBlock(2));
   P_W->MultTranspose(trueRhs.GetBlock(3), finalRhs.GetBlock(3));
   if (direct_solver)
       DirectSolver->Mult(finalRhs, trueX);
   else
   {
       solver.SetPrintLevel(0);
       solver.Mult(finalRhs, trueX);
   }

   chrono.Stop();


   if (verbose && !direct_solver) // iterative solver reports about its convergence
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

   if (strcmp(sigmaspace,"Hdiv") == 0)
   {
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
   }
   else
       if (verbose)
           std::cout << "Computation of the error in Hdiv norm is not implemented"
                        " when vector H1 space is used for sigma \n";

   // Computing error for S

   double err_S = S->ComputeL2Error((*Mytest.scalarS), irs);
   double norm_S = ComputeGlobalLpNorm(2, (*Mytest.scalarS), *pmesh, irs);
   if (verbose)
   {
       std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                    err_S / norm_S << "\n";
   }

   {
       FiniteElementCollection * hcurl_coll;
       if(dim==4)
           hcurl_coll = new ND1_4DFECollection;
       else
           hcurl_coll = new ND_FECollection(feorder+1, dim);
       auto *N_space = new ParFiniteElementSpace(pmesh.get(), hcurl_coll);

       DiscreteLinearOperator Grad(H_space, N_space);
       Grad.AddDomainInterpolator(new GradientInterpolator());
       ParGridFunction GradS(N_space);
       Grad.Assemble();
       Grad.Mult(*S, GradS);

       if (numsol != -34 && verbose)
           std::cout << "For this norm we are grad S for S from numsol = -34 \n";
       VectorFunctionCoefficient GradS_coeff(dim, uFunTest_ex_gradxt);
       double err_GradS = GradS.ComputeL2Error(GradS_coeff, irs);
       double norm_GradS = ComputeGlobalLpNorm(2, GradS_coeff, *pmesh, irs);
       if (verbose)
       {
           std::cout << "|| Grad_h (S_h - S_ex) || / || Grad S_ex || = " <<
                        err_GradS / norm_GradS << "\n";
           std::cout << "|| S_h - S_ex ||_H^1 / || S_ex ||_H^1 = " <<
                        sqrt(err_S*err_S + err_GradS*err_GradS) / sqrt(norm_S*norm_S + norm_GradS*norm_GradS) << "\n";
       }

       delete hcurl_coll;
       delete N_space;
   }

   /*
   // Check value of functional and mass conservation
   {
       double localFunctional = -2.0*(trueX.GetBlock(1)*trueRhs.GetBlock(1));

       trueX.GetBlock(2) = 0.0;
       trueRhs = 0.0;;
       CFOSLSop->Mult(trueX, trueRhs);
       localFunctional += trueX*(trueRhs);

       double globalFunctional;
       MPI_Reduce(&localFunctional, &globalFunctional, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
       if (verbose)
       {
           cout << "|| sigma_h - L(S_h) ||^2 + || div_h (bS_h) - f ||^2 = " << globalFunctional+norm_div*norm_div<< "\n";
           cout << "|| f ||^2 = " << norm_div*norm_div  << "\n";
           cout << "Relative Energy Error = " << sqrt(globalFunctional+norm_div*norm_div)/norm_div<< "\n";
       }

       ParLinearForm massform(W_space);
       massform.AddDomainIntegrator(new DomainLFIntegrator(*(Mytest.scalardivsigma)));
       massform.Assemble();

       double mass_loc = massform.Norml1();
       double mass;
       MPI_Reduce(&mass_loc, &mass, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
       if (verbose)
           cout << "Sum of local mass = " << mass<< "\n";

       trueRhs.GetBlock(2) -= massform;
       double mass_loss_loc = trueRhs.GetBlock(2).Norml1();
       double mass_loss;
       MPI_Reduce(&mass_loss_loc, &mass_loss, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
       if (verbose)
           cout << "Sum of local mass loss = " << mass_loss<< "\n";
   }
   */

   if (verbose)
       cout << "Computing projection errors" << endl;

   double projection_error_sigma = sigma_exact->ComputeL2Error(*(Mytest.sigma), irs);

   if(verbose)
   {
       cout << "|| sigma_ex - Pi_h sigma_ex || / || sigma_ex || = "
                       << projection_error_sigma / norm_sigma << endl;
   }

   double projection_error_S = S_exact->ComputeL2Error(*(Mytest.scalarS), irs);

   if(verbose)
       cout << "|| S_ex - Pi_h S_ex || / || S_ex || = "
                       << projection_error_S / norm_S << endl;


   if (visualization && nDimensions < 4)
   {
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

      *sigma_exact -= *sigma;

      socketstream uuu_sock(vishost, visport);
      uuu_sock << "parallel " << num_procs << " " << myid << "\n";
      uuu_sock.precision(8);
      uuu_sock << "solution\n" << *pmesh << *sigma_exact << "window_title 'difference for sigma'"
             << endl;

      socketstream s_sock(vishost, visport);
      s_sock << "parallel " << num_procs << " " << myid << "\n";
      s_sock.precision(8);
      MPI_Barrier(pmesh->GetComm());
      s_sock << "solution\n" << *pmesh << *S_exact << "window_title 'S_exact'"
              << endl;

      socketstream ss_sock(vishost, visport);
      ss_sock << "parallel " << num_procs << " " << myid << "\n";
      ss_sock.precision(8);
      MPI_Barrier(pmesh->GetComm());
      ss_sock << "solution\n" << *pmesh << *S << "window_title 'S'"
              << endl;

      *S_exact -= *S;
      socketstream sss_sock(vishost, visport);
      sss_sock << "parallel " << num_procs << " " << myid << "\n";
      sss_sock.precision(8);
      MPI_Barrier(pmesh->GetComm());
      sss_sock << "solution\n" << *pmesh << *S_exact
               << "window_title 'difference for S'" << endl;

      MPI_Barrier(pmesh->GetComm());
   }

   // 17. Free the used memory.
   //delete rhssig_form;
   //delete CFOSLSop;
   //delete A;

   //delete Asig_block;
   delete W_space;
   delete R_space;
   delete l2_coll;
   delete h1_coll;
   delete sigma_coll;

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
//    int nDimensions = xt.Size();
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
//    double t = xt(xt.Size()-1);
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
//    double x = xt(0);
//    double y = xt(1);
//    double t = xt(xt.Size()-1);
    return 0.0;
}

void uFun5_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
//    double t = xt(xt.Size()-1);

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
//    double x = xt(0);
//    double y = xt(1);
//    double t = xt(xt.Size()-1);
    return -10.0 * uFun6_ex(xt);
}

void uFun6_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
//    double t = xt(xt.Size()-1);

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
//    double x = xt(0);
//    double y = xt(1);
//    double t = xt(xt.Size()-1);

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
//    double x = xt(0);
//    double y = xt(1);
//    double z = xt(2);
//    double t = xt(xt.Size()-1);
    return -10.0 * uFun6_ex(xt);
}

void uFun66_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
//    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = -100.0 * 2.0 * (x - 0.5)  * uFun6_ex(xt);
    gradx(1) = -100.0 * 2.0 * y          * uFun6_ex(xt);
    gradx(2) = -100.0 * 2.0 * (z - 0.25) * uFun6_ex(xt);
}

void Hdivtest_fun(const Vector& xt, Vector& out )
{
    out.SetSize(xt.Size());

    double x = xt(0);
//    double y = xt(1);
//    double z = xt(2);
//    double t = xt(xt.Size()-1);

    out(0) = x;
    out(1) = 0.0;
    out(2) = 0.0;
    out(xt.Size()-1) = 0.;

}

double L2test_fun(const Vector& xt)
{
    double x = xt(0);
//    double y = xt(1);
//    double z = xt(2);
//    double t = xt(xt.Size()-1);

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
//    double z = xt(2);
    double t = xt(xt.Size()-1);
    return (cos(t)*exp(t) + sin(t)*exp(t)) * x * y;
}

void uFun10_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
//    double z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = sin(t)*exp(t)*y;
    gradx(1) = sin(t)*exp(t)*x;
    gradx(2) = 0.0;
}

