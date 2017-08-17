//
//                        MFEM CFOSLS Heat equation with multilevel algorithm and multigrid (div-free part)
//

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

class DivPart
{

public:

// Returns the particular solution, sigma
void div_part( int ref_levels,
               SparseMatrix *M_fine,
               SparseMatrix *B_fine,
               Vector &G_fine,
               Vector &F_fine,
               Array< SparseMatrix*> &P_W,
               Array< SparseMatrix*> &P_R,
               Array< SparseMatrix*> &Element_Elementc,
               Array< SparseMatrix*> &Element_dofs_R,
               Array< SparseMatrix*> &Element_dofs_W,
               HypreParMatrix * d_td_coarse_R,
               HypreParMatrix * d_td_coarse_W,
               Vector &sigma, Array<int>& ess_dof_coarsestlvl_list
              )
{
    StopWatch chrono;

    Vector sol_p_c2f;
    Vector vec1;

    Vector rhs_l;
    Vector comp;
    Vector F_coarse;


    Vector total_sig(P_R[0]->Height());
    total_sig = .0;

    chrono.Clear();
    chrono.Start();

    for (int l=0; l < ref_levels; l++)
         {
               // 1. Obtaining the relation Dofs_Coarse_Element
           SparseMatrix *R_t = Transpose(*Element_dofs_R[l]);
           SparseMatrix *W_t = Transpose(*Element_dofs_W[l]);

           MFEM_ASSERT(R_t->Width() == Element_Elementc[l]->Height() ,
                       "Element_Elementc matrix and R_t does not match");

           SparseMatrix *W_AE = Mult(*W_t,*Element_Elementc[l]);
           SparseMatrix *R_AE = Mult(*R_t,*Element_Elementc[l]);

           // 2. For RT elements, we impose boundary condition equal zero,
           //   see the function: GetInternalDofs2AE to obtained them

           SparseMatrix intDofs_R_AE;
           GetInternalDofs2AE(*R_AE,intDofs_R_AE);

           //  AE elements x localDofs stored in AE_R & AE_W
           SparseMatrix *AE_R =  Transpose(intDofs_R_AE);
           SparseMatrix *AE_W = Transpose(*W_AE);


           // 3. Right hand size at each level is of the form:
           //
           //   rhs = F - (P_W[l])^T inv((P_W[l]^T)(P_W[l]))(P_W^T)F

           rhs_l.SetSize(P_W[l]->Height());

           if(l ==0)
                rhs_l = F_fine;

           if (l>0)
               rhs_l = comp;

           comp.SetSize(P_W[l]->Width());

           F_coarse.SetSize(P_W[l]->Height());

           P_W[l]->MultTranspose(rhs_l,comp);

           SparseMatrix * P_WT = Transpose(*P_W[l]);
           SparseMatrix * P_WTxP_W = Mult(*P_WT,*P_W[l]);
           Vector Diag(P_WTxP_W->Size());
           Vector invDiag(P_WTxP_W->Size());
           P_WTxP_W->GetDiag(Diag);

           for(int m=0; m < P_WTxP_W->Size(); m++)
              invDiag(m) = comp(m)/Diag(m);


           P_W[l]->Mult(invDiag,F_coarse);



           rhs_l -=F_coarse;

           MFEM_ASSERT(rhs_l.Sum()<= 9e-11,
                       "Average of rhs at each level is not zero: " << rhs_l.Sum());


           if (l> 0) {

              // 4. Creating matrices for the coarse problem:
               SparseMatrix *P_WT2 = Transpose(*P_W[l-1]);
               SparseMatrix *P_RT2 = Transpose(*P_R[l-1]);

               SparseMatrix *B_PR = Mult(*B_fine, *P_R[l-1]);
               B_fine = Mult(*P_WT2, *B_PR);

               SparseMatrix *M_PR = Mult(*M_fine, *P_R[l-1]);
               M_fine = Mult(*P_RT2, *M_PR);

             }

           //5. Setting for the coarse problem
           DenseMatrix sub_M;
           DenseMatrix sub_B;
           DenseMatrix sub_BT;
           DenseMatrix invBB;

           Vector sub_F;
           Vector sub_G;

           //Vector to Assamble the solution at level l
           Vector u_loc_vec(AE_W->Width());
           Vector p_loc_vec(AE_R->Width());

           u_loc_vec =0.0;
           p_loc_vec =0.0;


         for( int e = 0; e < AE_R->Height(); e++){

             Array<int> Rtmp_j(AE_R->GetRowColumns(e), AE_R->RowSize(e));
             Array<int> Wtmp_j(AE_W->GetRowColumns(e), AE_W->RowSize(e));

             // Setting size of Dense Matrices
             sub_M.SetSize(Rtmp_j.Size());
             sub_B.SetSize(Wtmp_j.Size(),Rtmp_j.Size());
             sub_BT.SetSize(Rtmp_j.Size(),Wtmp_j.Size());
             sub_G.SetSize(Rtmp_j.Size());
             sub_F.SetSize(Wtmp_j.Size());

             // Obtaining submatrices:
             M_fine->GetSubMatrix(Rtmp_j,Rtmp_j, sub_M);
             B_fine->GetSubMatrix(Wtmp_j,Rtmp_j, sub_B);
             sub_BT.Transpose(sub_B);

             sub_G  = .0;
             sub_F  = .0;

             rhs_l.GetSubVector(Wtmp_j, sub_F);


             Vector sig(Rtmp_j.Size());

            if (e ==1){
             MFEM_ASSERT(sub_F.Sum()<= 9e-11,
                       "checking global average at each level " << sub_F.Sum());
          }
             Vector sub_FF = sub_F;

            // Solving local problem:
             Local_problem(sub_M, sub_B, sub_G, sub_F,sig);

         if ( e ==1){

             // Checking if the local problems satisfy the condition
             Vector fcheck(Wtmp_j.Size());
             fcheck =.0;
             sub_B.Mult(sig, fcheck);
             fcheck-=sub_FF;
             MFEM_ASSERT(fcheck.Norml2()<= 9e-11,
                       "checking global average at each level " << fcheck.Norml2());


             }

         p_loc_vec.AddElementVector(Rtmp_j,sig);

         }

           Vector fcheck2(u_loc_vec.Size());
           fcheck2 = .0;
           B_fine->Mult(p_loc_vec, fcheck2);
           fcheck2-=rhs_l;
           MFEM_ASSERT(fcheck2.Norml2()<= 9e-11,
                       "checking global solution at each level " << fcheck2.Norml2());


            // Final Solution ==
            if (l>0){
              for (int k = l-1; k>=0; k--){

                vec1.SetSize(P_R[k]->Height());
                P_R[k]->Mult(p_loc_vec, vec1);
                p_loc_vec = vec1;

              }
            }

            total_sig +=p_loc_vec;

            MFEM_ASSERT(fcheck2.Norml2()<= 9e+9,
                       "checking global solution added" << total_sig.Norml2());

        }


     // The coarse problem::

         SparseMatrix *M_coarse;
         SparseMatrix *B_coarse;
         Vector FF_coarse(P_W[ref_levels-1]->Width());

         rhs_l +=F_coarse;
         P_W[ref_levels-1]->MultTranspose(rhs_l, FF_coarse );

         SparseMatrix *P_WT2 = Transpose(*P_W[ref_levels-1]);
             SparseMatrix *P_RT2 = Transpose(*P_R[ref_levels-1]);

             SparseMatrix *B_PR = Mult(*B_fine, *P_R[ref_levels-1]);
             B_coarse = Mult(*P_WT2, *B_PR);

             B_coarse->EliminateCols(ess_dof_coarsestlvl_list);

             SparseMatrix *M_PR = Mult(*M_fine, *P_R[ref_levels-1]);

             M_coarse =  Mult(*P_RT2, *M_PR);
             //std::cout << "M_coarse size = " << M_coarse->Height() << "\n";
             for ( int k = 0; k < ess_dof_coarsestlvl_list.Size(); ++k)
                 if (ess_dof_coarsestlvl_list[k] !=0)
                        M_coarse->EliminateRowCol(k);

             Vector sig_c(B_coarse->Width());

             auto d_td_M = d_td_coarse_R->LeftDiagMult(*M_coarse);
             HypreParMatrix *d_td_T = d_td_coarse_R->Transpose();

             HypreParMatrix *M_Global = ParMult(d_td_T, d_td_M);

             auto B_Global = d_td_coarse_R->LeftDiagMult(*B_coarse,d_td_coarse_W->GetColStarts());
             HypreParMatrix *BT = B_Global->Transpose();

         Vector Truesig_c(B_Global->Width());

       Array<int> block_offsets(3); // number of variables + 1
       block_offsets[0] = 0;
       block_offsets[1] = M_Global->Width();
       block_offsets[2] = B_Global->Height();
       block_offsets.PartialSum();

       BlockOperator coarseMatrix(block_offsets);
       coarseMatrix.SetBlock(0,0, M_Global);
       coarseMatrix.SetBlock(0,1, BT);
       coarseMatrix.SetBlock(1,0, B_Global);


       BlockVector trueX(block_offsets), trueRhs(block_offsets);
       trueRhs =0;
       trueRhs.GetBlock(1)= FF_coarse;


       // 9. Construct the operators for preconditioner
       //
       //                 P = [ diag(M)         0         ]
       //                     [  0       B diag(M)^-1 B^T ]
       //
       //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
       //     pressure Schur Complement

       HypreParMatrix *MinvBt = B_Global->Transpose();
       HypreParVector *Md = new HypreParVector(MPI_COMM_WORLD, M_Global->GetGlobalNumRows(),
                                               M_Global->GetRowStarts());
       M_Global->GetDiag(*Md);

       MinvBt->InvScaleRows(*Md);
       HypreParMatrix *S = ParMult(B_Global, MinvBt);

       //HypreSolver *invM, *invS;
       auto invM = new HypreDiagScale(*M_Global);
       auto invS = new HypreBoomerAMG(*S);
       invS->SetPrintLevel(0);
          invM->iterative_mode = false;
       invS->iterative_mode = false;

       BlockDiagonalPreconditioner *darcyPr = new BlockDiagonalPreconditioner(
          block_offsets);
       darcyPr->SetDiagonalBlock(0, invM);
       darcyPr->SetDiagonalBlock(1, invS);

       // 12. Solve the linear system with MINRES.
       //     Check the norm of the unpreconditioned residual.

       int maxIter(50000);
       double rtol(1.e-18);
       double atol(1.e-18);


       MINRESSolver solver(MPI_COMM_WORLD);
       solver.SetAbsTol(atol);
       solver.SetRelTol(rtol);
       solver.SetMaxIter(maxIter);
       solver.SetOperator(coarseMatrix);
       solver.SetPreconditioner(*darcyPr);
       solver.SetPrintLevel(0);
       trueX = 0.0;
       solver.Mult(trueRhs, trueX);
       chrono.Stop();

       //cout << "CG converged in " << solver.GetNumIterations() << " iterations" <<endl;
       cout << "MINRES solver took " << chrono.RealTime() << "s. \n";

       Truesig_c = trueX.GetBlock(0);

       d_td_coarse_R->Mult(Truesig_c,sig_c);
      for (int k = ref_levels-1; k>=0; k--){

                vec1.SetSize(P_R[k]->Height());
                P_R[k]->Mult(sig_c, vec1);
                sig_c.SetSize(P_R[k]->Height());
                sig_c = vec1;

            }

       total_sig+=sig_c;
       sigma.SetSize(total_sig.Size());
       sigma = total_sig;
}

void Dofs_AE(SparseMatrix &Element_Dofs, const SparseMatrix &Element_Element_coarse, SparseMatrix &Dofs_Ae)
{
        // Returns a SparseMatrix with the relation dofs to Element coarse.
        SparseMatrix *R_T = Transpose(Element_Dofs);
        SparseMatrix *Dofs_AE = Mult(*R_T,Element_Element_coarse);
        SparseMatrix *AeDofs = Transpose(*Dofs_AE);
        Dofs_Ae = *AeDofs;
}


void Elem2Dofs(const FiniteElementSpace &fes, SparseMatrix &Element_to_dofs)
{
  // Returns a SparseMatrix with the relation Element to Dofs
  int * I = new int[fes.GetNE()+1];
  Array<int> vdofs_R;
  Array<int> dofs_R;

  I[0] = 0;
  for (int i = 0; i < fes.GetNE(); i++)
    {
      fes.GetElementVDofs(i, vdofs_R);
      I[i+1] = I[i] + vdofs_R.Size();
    }
  int * J = new int[I[fes.GetNE()]];
  double * data = new double[I[fes.GetNE()]];

  for (int i = 0; i<fes.GetNE(); i++)
    {
      // Returns indexes of dofs in array for ith' elements'
      fes.GetElementVDofs(i,vdofs_R);
      fes.AdjustVDofs(vdofs_R);
      for (int j = I[i];j<I[i+1];j++)
        {
          J[j] = vdofs_R[j-I[i]];
          data[j] =1;
        }

    }
  SparseMatrix A(I,J,data,fes.GetNE(), fes.GetVSize());
  Element_to_dofs.Swap(A);
}

void GetInternalDofs2AE(const SparseMatrix &R_AE, SparseMatrix &B)
{
  /* Returns a SparseMatrix with the relation InteriorDofs to Coarse Element.
   * This is use for the Raviart-Thomas dofs, which vanish at the
   * boundary of the coarse elements.
   *
   * row.Size() ==2, means, it share by 2 AE
   *
   * For the lowest order case:
   * row.Size()==1, and data=1, means bdry
   * row.Size()==1, and data=2, means interior
   */

  int nnz=0;
  int * R_AE_i = R_AE.GetI();
  int * R_AE_j = R_AE.GetJ();
  double * R_AE_data = R_AE.GetData();

  int * out_i = new int [R_AE.Height()+1];

  // Find Hdivdofs_interior_AE
  for (int i=0; i<R_AE.Height(); i++)
    {
      out_i[i]= nnz;
      for (int j= R_AE_i[i]; j< R_AE_i[i+1]; j++)
        if (R_AE_data[j]==2)
           nnz++; // If the degree is share by two elements
    }
  out_i[R_AE.Height()] = nnz;

  int * out_j = new int[nnz];
  double * out_data = new double[nnz];
  nnz = 0;

  for (int i=0; i< R_AE.Height(); i++)
    for (int j=R_AE_i[i]; j<R_AE_i[i+1]; j++)
      if (R_AE_data[j] == 2)
        out_j[nnz++] = R_AE_j[j];

  // Forming the data array:
  std::fill_n(out_data, nnz, 1);

  SparseMatrix out(out_i, out_j, out_data, R_AE.Height(),
                   R_AE.Width());
  B.Swap(out);
}

void Local_problem(const DenseMatrix &sub_M,  DenseMatrix &sub_B, Vector &Sub_G, Vector &sub_F, Vector &sigma){
                  // Returns sigma local

                  DenseMatrixInverse invM_loc(sub_M);
                  DenseMatrix sub_BT(sub_B.Width(), sub_B.Height());
                  sub_BT.Transpose(sub_B);

                  DenseMatrix invM_BT(sub_B.Width());
          invM_loc.Mult(sub_BT,invM_BT);


          /* Solving the local problem:
                  *
              * Msig + B^tu = G
              * Bsig        = F
              *
              * sig =  M^{-1} B^t(-u) + M^{-1} G
              *
              * B M^{-1} B^t (-u) = F
              */

          DenseMatrix B_invM_BT;
          B_invM_BT = 0.0;
          B_invM_BT.SetSize(sub_B.Height());

          Mult(sub_B, invM_BT, B_invM_BT);

          Vector one(sub_B.Height());
          one = 0.0;
          one[0] =1;
          B_invM_BT.SetRow(0,0);
          B_invM_BT.SetCol(0,0);
          B_invM_BT.SetCol(0,one);
          B_invM_BT(0,0)=1;


          DenseMatrixInverse inv_BinvMBT(B_invM_BT);

          Vector invMG(sub_M.Size());
          invM_loc.Mult(Sub_G,invMG);

          sub_F[0] = 0;
          Vector uu(sub_B.Height());
          inv_BinvMBT.Mult(sub_F, uu);
          invM_BT.Mult(uu,sigma);
          sigma += invMG;
}

};

class VectorcurlDomainLFIntegrator : public LinearFormIntegrator
{
    DenseMatrix curlshape;
    DenseMatrix curlshape_dFadj;
    DenseMatrix curlshape_dFT;
    DenseMatrix dF_curlshape;
    VectorCoefficient &VQ;
    int oa, ob;
public:
    /// Constructs a domain integrator with a given Coefficient
    VectorcurlDomainLFIntegrator(VectorCoefficient &VQF, int a = 2, int b = 0)
        : VQ(VQF), oa(a), ob(b) { }

    /// Constructs a domain integrator with a given Coefficient
    VectorcurlDomainLFIntegrator(VectorCoefficient &VQF, const IntegrationRule *ir)
        : LinearFormIntegrator(ir), VQ(VQF), oa(1), ob(1) { }

    /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect);

    using LinearFormIntegrator::AssembleRHSElementVect;
};

void VectorcurlDomainLFIntegrator::AssembleRHSElementVect(
        const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
    int dof = el.GetDof();

    int dim = el.GetDim();
    MFEM_ASSERT(dim == 3, "VectorcurlDomainLFIntegrator is working only in 3D currently \n");

    curlshape.SetSize(dof,3);           // matrix of size dof x 3, works only in 3D
    curlshape_dFadj.SetSize(dof,3);     // matrix of size dof x 3, works only in 3D
    curlshape_dFT.SetSize(dof,3);       // matrix of size dof x 3, works only in 3D
    dF_curlshape.SetSize(3,dof);        // matrix of size dof x 3, works only in 3D
    Vector vecval(3);
    //Vector vecval_new(3);
    //DenseMatrix invdfdx(3,3);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        // ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob + Tr.OrderW());
        ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
        // int order = 2 * el.GetOrder() ; // <--- OK for RTk
        // ir = &IntRules.Get(el.GetGeomType(), order);
    }

    elvect.SetSize(dof);
    elvect = 0.0;

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        el.CalcCurlShape(ip, curlshape);

        Tr.SetIntPoint (&ip);

        VQ.Eval(vecval,Tr,ip);                  // plain evaluation

        MultABt(curlshape, Tr.Jacobian(), curlshape_dFT);

        curlshape_dFT.AddMult_a(ip.weight, vecval, elvect);
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

       //double val = Tr.Weight() * Q.Eval(Tr, ip);

       Tr.SetIntPoint (&ip);
       w = ip.weight;// * Tr.Weight();
       CalcAdjugate(Tr.Jacobian(), invdfdx);
       Mult(dshape, invdfdx, dshapedxt);

       Q.Eval(bf, Tr, ip);

       dshapedxt.Mult(bf, bfdshapedxt);

       add(elvect, w, bfdshapedxt, elvect);
    }
}

/** Bilinear integrator for (curl u, v) for Nedelec and scalar finite element for v. If the trial and
    test spaces are switched, assembles the form (u, curl v). */
/*
class VectorFECurlVQIntegrator: public BilinearFormIntegrator
{
private:
   VectorCoefficient *VQ;
#ifndef MFEM_THREAD_SAFE
   Vector shape;
   DenseMatrix curlshape;
   DenseMatrix curlshape_dFT;
   //old
   DenseMatrix curlshapeTrial;
   DenseMatrix vshapeTest;
   DenseMatrix curlshapeTrial_dFT;
#endif
   void Init(VectorCoefficient *vq)
   { VQ = vq; }
public:
   VectorFECurlVQIntegrator() { Init(NULL); }
   VectorFECurlVQIntegrator(VectorCoefficient &vq) { Init(&vq); }
   VectorFECurlVQIntegrator(VectorCoefficient *vq) { Init(vq); }
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat) { }
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

void VectorFECurlVQIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int trial_nd = trial_fe.GetDof(), test_nd = test_fe.GetDof(), i;
   //int dim = trial_fe.GetDim();
   //int dimc = (dim == 3) ? 3 : 1;
   int dim;
   int vector_dof, scalar_dof;

   MFEM_ASSERT(trial_fe.GetMapType() == mfem::FiniteElement::H_CURL ||
               test_fe.GetMapType() == mfem::FiniteElement::H_CURL,
               "At least one of the finite elements must be in H(Curl)");

   //int curl_nd;
   int vec_nd;
   if ( trial_fe.GetMapType() == mfem::FiniteElement::H_CURL )
   {
      //curl_nd = trial_nd;
      vector_dof = trial_fe.GetDof();
      vec_nd  = test_nd;
      scalar_dof = test_fe.GetDof();
      dim = trial_fe.GetDim();
   }
   else
   {
      //curl_nd = test_nd;
      vector_dof = test_fe.GetDof();
      vec_nd  = trial_nd;
      scalar_dof = trial_fe.GetDof();
      dim = test_fe.GetDim();
   }

   MFEM_ASSERT(dim == 3, "VectorFECurlVQIntegrator is working only in 3D currently \n");

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshapeTrial(curl_nd, dimc);
   DenseMatrix curlshapeTrial_dFT(curl_nd, dimc);
   DenseMatrix vshapeTest(vec_nd, dimc);
#else
   //curlshapeTrial.SetSize(curl_nd, dimc);
   //curlshapeTrial_dFT.SetSize(curl_nd, dimc);
   //vshapeTest.SetSize(vec_nd, dimc);
#endif
   //Vector shapeTest(vshapeTest.GetData(), vec_nd);

   curlshape.SetSize(vector_dof, dim);
   curlshape_dFT.SetSize(vector_dof, dim);
   shape.SetSize(scalar_dof);
   Vector D(vec_nd);

   elmat.SetSize(test_nd, trial_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() - 1; // <--
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   elmat = 0.0;
   for (i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Trans.SetIntPoint(&ip);

      double w = ip.weight;
      VQ->Eval(D, Trans, ip);
      D *= w;

      if (dim == 3)
      {
         if ( trial_fe.GetMapType() == mfem::FiniteElement::H_CURL )
         {
            trial_fe.CalcCurlShape(ip, curlshape);
            test_fe.CalcShape(ip, shape);
         }
         else
         {
            test_fe.CalcCurlShape(ip, curlshape);
            trial_fe.CalcShape(ip, shape);
         }
         MultABt(curlshape, Trans.Jacobian(), curlshape_dFT);

         ///////////////////////////
         for (int d = 0; d < dim; d++)
         {
            for (int j = 0; j < scalar_dof; j++)
            {
                for (int k = 0; k < vector_dof; k++)
                {
                    elmat(j, k) += D[d] * shape(j) * curlshape_dFT(k, d);
                }
            }
         }
         ///////////////////////////
      }
   }
}
*/

//////////////////// new
class HeatVectorFECurlSigmaIntegrator: public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
    Vector shape;
    DenseMatrix dshape;
    DenseMatrix curlshape;
    DenseMatrix curlshape_dFT;
    DenseMatrix dshapedxt;
    DenseMatrix invdfdx;
    //old
    DenseMatrix curlshapeTrial;
    DenseMatrix vshapeTest;
    DenseMatrix curlshapeTrial_dFT;
#endif
public:
    HeatVectorFECurlSigmaIntegrator() {}
    virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat) { }
    virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

void HeatVectorFECurlSigmaIntegrator::AssembleElementMatrix2(
const FiniteElement &trial_fe, const FiniteElement &test_fe,
ElementTransformation &Trans, DenseMatrix &elmat)
{
    int trial_nd = trial_fe.GetDof(), test_nd = test_fe.GetDof(), i;

    int dim;
    int vector_dof, scalar_dof;

    MFEM_ASSERT(trial_fe.GetMapType() == mfem::FiniteElement::H_CURL ||
               test_fe.GetMapType() == mfem::FiniteElement::H_CURL,
               "At least one of the finite elements must be in H(Curl)");

    //int curl_nd;
    //int vec_nd;
    if ( trial_fe.GetMapType() == mfem::FiniteElement::H_CURL )
    {
      //curl_nd = trial_nd;
      vector_dof = trial_fe.GetDof();
      //vec_nd  = test_nd;
      scalar_dof = test_fe.GetDof();
      dim = trial_fe.GetDim();
    }
    else
    {
      //curl_nd = test_nd;
      vector_dof = test_fe.GetDof();
      //vec_nd  = trial_nd;
      scalar_dof = trial_fe.GetDof();
      dim = test_fe.GetDim();
    }

    MFEM_ASSERT(dim == 3, "HeatVectorFECurlSigmaIntegrator is working only in 3D currently \n");

    #ifdef MFEM_THREAD_SAFE
    DenseMatrix curlshapeTrial(curl_nd, dimc);
    DenseMatrix curlshapeTrial_dFT(curl_nd, dimc);
    DenseMatrix vshapeTest(vec_nd, dimc);
    DenseMatrix dshape(scalar_dof,dim);
    DenseMatrix dshapedxt(scalar_dof,dim);
    DenseMatrix invdfdx(dim,dim);
    #else
    //curlshapeTrial.SetSize(curl_nd, dimc);
    //curlshapeTrial_dFT.SetSize(curl_nd, dimc);
    //vshapeTest.SetSize(vec_nd, dimc);
    #endif
    //Vector shapeTest(vshapeTest.GetData(), vec_nd);

    curlshape.SetSize(vector_dof, dim);
    curlshape_dFT.SetSize(vector_dof, dim);
    shape.SetSize(scalar_dof);
    dshape.SetSize(scalar_dof,dim);
    dshapedxt.SetSize(scalar_dof,dim);
    invdfdx.SetSize(dim,dim);

    elmat.SetSize(test_nd, trial_nd);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() - 1; // <--
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
    }

    elmat = 0.0;
    for (i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        Trans.SetIntPoint(&ip);

        double w = ip.weight * Trans.Weight();

        if (dim == 3)
        {
            if ( trial_fe.GetMapType() == mfem::FiniteElement::H_CURL )
            {
               trial_fe.CalcCurlShape(ip, curlshape);
               test_fe.CalcShape(ip, shape);
               test_fe.CalcDShape(ip, dshape);
            }
            else
            {
               test_fe.CalcCurlShape(ip, curlshape);
               trial_fe.CalcShape(ip, shape);
               trial_fe.CalcDShape(ip, dshape);
            }
            CalcInverse(Trans.Jacobian(), invdfdx);
            Mult(dshape, invdfdx, dshapedxt);

            MultABt(curlshape, Trans.Jacobian(), curlshape_dFT);

            ///////////////////////////
            for (int d = 0; d < dim; d++)
            {
               for (int j = 0; j < scalar_dof; j++)
               {
                   for (int k = 0; k < vector_dof; k++)
                   {
                       // last component: instead of d scalar_phi_i / dt
                       // we use - scalar_phi
                       if (d == dim - 1)
                           elmat(j, k) += w * (-shape(j)) * curlshape_dFT(k, d);
                       else
                           elmat(j, k) += w * dshapedxt(j,d) * curlshape_dFT(k, d);
                   }
               }
            }
            ///////////////////////////
         } // end of if dim = 3
    } // end of loop over integration points
}
class HeatSigmaLFIntegrator: public LinearFormIntegrator
{
    Vector el_shape;
    DenseMatrix el_dshape;
    DenseMatrix invdfdx;
    DenseMatrix el_dshapedxt;
    Vector coeffvec;
    Vector local_vec;
    VectorCoefficient &Q;
    int oa, ob;
 public:
    /// Constructs a domain integrator with a given Coefficient
    HeatSigmaLFIntegrator(VectorCoefficient &QF, int a = 2, int b = 0)
    // the old default was a = 1, b = 1
    // for simple elliptic problems a = 2, b = -2 is ok
       : Q(QF), oa(a), ob(b) { }

    /// Constructs a domain integrator with a given Coefficient
    HeatSigmaLFIntegrator(VectorCoefficient &QF, const IntegrationRule *ir)
       : LinearFormIntegrator(ir), Q(QF), oa(1), ob(1) { }

    /** Given a particular Finite Element and a transformation (Tr)
        computes the element right hand side element vector, elvect. */
    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect);

    using LinearFormIntegrator::AssembleRHSElementVect;
};

void HeatSigmaLFIntegrator::AssembleRHSElementVect(
    const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
    int dof = el.GetDof();
    int dim  = el.GetDim();

    el_shape.SetSize(dof);
    el_dshape.SetSize(dof,dim);
    invdfdx.SetSize(dim,dim);
    el_dshapedxt.SetSize(dof,dim);
    local_vec.SetSize(dof);

    elvect.SetSize(dof);
    elvect = 0.0;

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
       Tr.SetIntPoint (&ip);

       el.CalcShape(ip, el_shape);
       el.CalcDShape(ip, el_dshape);

       w = ip.weight * Tr.Weight();
       CalcInverse(Tr.Jacobian(), invdfdx);
       // dshapedxt = matrix d phi_i / dx^j ~ (dof,dim)
       Mult(el_dshape, invdfdx, el_dshapedxt);

       // replacing last column (related to t) of dshapedxt by (-phi_i)
       for (int dofind = 0; dofind < dof; ++dofind)
           el_dshapedxt(dofind, dim - 1) = - el_shape(dofind);

       // coeffvec = values of vector coefficient at dofs
       Q.Eval(coeffvec, Tr, ip);

       el_dshapedxt.Mult(coeffvec, local_vec);
       add(elvect, w, local_vec, elvect);
    }
}
//////////////////// new




//********* NEW STUFF FOR 4D HEAT CFOSLS
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
    Vector trial_shape;
    Vector test_shape;//<<<<<<<
    DenseMatrix K;
    DenseMatrix test_vshape;
    DenseMatrix trial_vshape;
    DenseMatrix trial_dshape;//<<<<<<<<<<<<<<
    DenseMatrix test_dshape;//<<<<<<<<<<<<<<

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
    void Init(Coefficient *q)
    { Q = q; }

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
    PAUVectorFEMassIntegrator2() { Init(NULL); }
    PAUVectorFEMassIntegrator2(Coefficient *_q) { Init(_q); }
    PAUVectorFEMassIntegrator2(Coefficient &q) { Init(&q); }

    virtual void AssembleElementMatrix(const FiniteElement &el,
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
    // assume both test_fe and trial_fe are vector FE
    int dim  = test_fe.GetDim();
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();
    double w;

    if (VQ || MQ) // || = or
        mfem_error("VectorFEMassIntegrator::AssembleElementMatrix2(...)\n"
                   "   is not implemented for vector/tensor permeability");

    DenseMatrix trial_dshapedxt(trial_dof,dim);
    DenseMatrix invdfdx(dim,dim);

#ifdef MFEM_THREAD_SAFE
    // DenseMatrix trial_vshape(trial_dof, dim);
    Vector trial_shape(trial_dof); //PAULI
    DenseMatrix trial_dshape(trial_dof,dim);
    DenseMatrix test_vshape(test_dof,dim);
#else
    //trial_vshape.SetSize(trial_dof, dim);
    trial_shape.SetSize(trial_dof); //PAULI
    trial_dshape.SetSize(trial_dof,dim); //Pauli
    test_vshape.SetSize(test_dof,dim);
#endif
    //elmat.SetSize (test_dof, trial_dof);
    elmat.SetSize (test_dof, trial_dof);

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

        trial_fe.CalcShape(ip, trial_shape);
        trial_fe.CalcDShape(ip, trial_dshape);

        Trans.SetIntPoint (&ip);
        test_fe.CalcVShape(Trans, test_vshape);

        w = ip.weight * Trans.Weight();
        CalcInverse(Trans.Jacobian(), invdfdx);
        Mult(trial_dshape, invdfdx, trial_dshapedxt);
        if (Q)
        {
            w *= Q -> Eval (Trans, ip);
        }

        for (int j = 0; j < test_dof; j++)
        {
            for (int k = 0; k < trial_dof; k++)
            {
                for (int d = 0; d < dim - 1; d++ )
                    elmat(j, k) += 1.0 * w * test_vshape(j, d) * trial_dshapedxt(k, d);
                elmat(j, k) -= w * test_vshape(j, dim - 1) * trial_shape(k);
            }
        }
    }
}

void PAUVectorFEMassIntegrator2::AssembleElementMatrix(
        const FiniteElement &el,
        ElementTransformation &Trans,
        DenseMatrix &elmat)
{
    int dof = el.GetDof();
    int dim  = el.GetDim();
    double w;

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
            for (int k = 0; k < dof; k++)
            {
                for (int d = 0; d < dim - 1; d++ )
                    elmat(j, k) +=  w * dshapedxt(j, d) * dshapedxt(k, d);
                elmat(j, k) +=  w * shape(j) * shape(k);
            }

    }
}

// Define the analytical solution and forcing terms / boundary conditions
//double u0_function(const Vector &x);
double uFun_ex(const Vector & x); // Exact Solution
double uFun_ex_dt(const Vector & xt);
double uFun_ex_laplace(const Vector & xt);
void uFun_ex_gradx(const Vector& xt, Vector& gradx );

double uFun1_ex(const Vector & x); // Exact Solution
double uFun1_ex_dt(const Vector & xt);
double uFun1_ex_laplace(const Vector & xt);
void uFun1_ex_gradx(const Vector& xt, Vector& gradx );

double uFun2_ex(const Vector & x); // Exact Solution
double uFun2_ex_dt(const Vector & xt);
double uFun2_ex_laplace(const Vector & xt);
void uFun2_ex_gradx(const Vector& xt, Vector& gradx );

double uFun3_ex(const Vector & x); // Exact Solution
double uFun3_ex_dt(const Vector & xt);
double uFun3_ex_laplace(const Vector & xt);
void uFun3_ex_gradx(const Vector& xt, Vector& gradx );

void hcurlFun3D_ex(const Vector& xt, Vector& vecvalue);
void curlhcurlFun3D_ex(const Vector& xt, Vector& vecvalue);
void hcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue);
void curlhcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue);

void DivmatFun4D_ex(const Vector& xt, Vector& vecvalue);
void DivmatDivmatFun4D_ex(const Vector& xt, Vector& vecvalue);

double zero_ex(const Vector& xt);
void zerovec_ex(const Vector& xt, Vector& vecvalue);
void zerovecx_ex(const Vector& xt, Vector& zerovecx );

void vminusone_exact(const Vector &x, Vector &vminusone);
void vone_exact(const Vector &x, Vector &vone);

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void curlE_exact(const Vector &x, Vector &curlE);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    double rhsideTemplate(const Vector& xt);

template<double (*S)(const Vector & xt)> \
    double SnonhomoTemplate(const Vector& xt);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    double divsigmaTemplate(const Vector& xt);


template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaTemplate(const Vector& xt, Vector& sigma);

template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaNonHomoTemplate(const Vector& xt, Vector& sigma);


template<double (*S)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),
        void (*opdivfreevec)(const Vector&, Vector& )>
        void sigmahatTemplate(const Vector& xt, Vector& sigmahatv);
template<double (*S)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),
        void (*opdivfreevec)(const Vector&, Vector& )>
        void minsigmahatTemplate(const Vector& xt, Vector& sigmahatv);


/*
class Transport_test_divfree
{
    protected:
        int dim;
        int numsol;
        int numcurl;

    public:
        FunctionCoefficient * scalarS;
        FunctionCoefficient * S_nonhomo;              // S_nonhomo(x,t) = S(x,t=0)
        FunctionCoefficient * scalarf;                // d (S - S_nonhomo) /dt + div (b [S - S_nonhomo]), Snonhomo = S(x,0)
        FunctionCoefficient * scalardivsigma;         // = dS/dt + div (bS) = div sigma
        FunctionCoefficient * bTb;                    // b^T * b
        FunctionCoefficient * minbTbSnonhomo;         // - b^T * b * S_nonhomo
        FunctionCoefficient * bsigmahat;              // b * sigma_hat
        VectorFunctionCoefficient * sigma;
        VectorFunctionCoefficient * sigmahat;         // sigma_hat = sigma_exact - op divfreepart (curl hcurlpart in 3D)
        VectorFunctionCoefficient * b;
        VectorFunctionCoefficient * minb;
        VectorFunctionCoefficient * bf;
        VectorFunctionCoefficient * bdivsigma;        // b * div sigma = b * initial f (before modifying it due to inhomogenuity)
        MatrixFunctionCoefficient * Ktilda;
        MatrixFunctionCoefficient * bbT;
        VectorFunctionCoefficient * sigma_nonhomo;    // to incorporate inhomogeneous boundary conditions, stores (b*S0, S0) with S(t=0) = S0
        VectorFunctionCoefficient * bSnonhomo;        // b * S_nonhomo
        VectorFunctionCoefficient * divfreepart;        // additional part added for testing div-free solver
        VectorFunctionCoefficient * opdivfreepart;    // curl of the additional part which is added to sigma_exact for testing div-free solver
        VectorFunctionCoefficient * minKsigmahat;     // - K * sigma_hat
        VectorFunctionCoefficient * minsigmahat;      // -sigma_hat
    public:
        Transport_test_divfree (int Dim, int NumSol, int NumCurl);

        int GetDim() {return dim;}
        int GetNumSol() {return numsol;}
        int GetNumCurl() {return numcurl;}
        void SetDim(int Dim) { dim = Dim;}
        void SetNumSol(int NumSol) { numsol = NumSol;}
        void SetNumCurl(int NumCurl) { numcurl = NumCurl;}
        bool CheckTestConfig();

        ~Transport_test_divfree () {}
    private:
        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbvec)(const Vector & xt), \
                 void(*hcurlvec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
        void SetTestCoeffs ( );

        void SetSFun( double (*S)(const Vector & xt))
        { scalarS = new FunctionCoefficient(S);}

        template< double (*S)(const Vector & xt)>  \
        void SetSNonhomo()
        {
            S_nonhomo = new FunctionCoefficient(SnonhomoTemplate<S>);
        }

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        void SetScalarfFun()
        { scalarf = new FunctionCoefficient(rhsideTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>);}

        template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec)>  \
        void SetminbTbSnonhomo()
        {
            minbTbSnonhomo = new FunctionCoefficient(minbTbSnonhomoTemplate<S,bvec>);
        }

        template<void(*bvec)(const Vector & x, Vector & vec)>  \
        void SetScalarBtB()
        {
            bTb = new FunctionCoefficient(bTbTemplate<bvec>);
        }

        template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec)> \
        void SetHdivVec()
        {
            sigma = new VectorFunctionCoefficient(dim, sigmaTemplate<S,bvec>);
        }

        template<void(*bvec)(const Vector & x, Vector & vec)> \
        void SetminbVec()
        { minb = new VectorFunctionCoefficient(dim, minbTemplate<bvec>);}

        void SetbVec( void(*bvec)(const Vector & x, Vector & vec) )
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

        template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec)> \
        void SetSigmaNonhomoVec()
        {
            sigma_nonhomo = new VectorFunctionCoefficient(dim, sigmaNonHomoTemplate<S,bvec>);
        }

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void SetdivSigma()
        { scalardivsigma = new FunctionCoefficient(divsigmaTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>);}

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        void SetbfVec()
        { bf = new VectorFunctionCoefficient(dim, bfTemplate<S,dSdt,Sgradxvec,bvec,divbfunc>);}

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        void SetbdivsigmaVec()
        { bdivsigma = new VectorFunctionCoefficient(dim, bdivsigmaTemplate<S,dSdt,Sgradxvec,bvec,divbfunc>);}

        void SetDivfreePart( void(*divfreevec)(const Vector & x, Vector & vec))
        { divfreepart = new VectorFunctionCoefficient(dim, divfreevec);}

        void SetOpDivfreePart( void(*opdivfreevec)(const Vector & x, Vector & vec))
        { opdivfreepart = new VectorFunctionCoefficient(dim, opdivfreevec);}

        template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
        void SetminKsigmahat()
        { minKsigmahat = new VectorFunctionCoefficient(dim, minKsigmahatTemplate<S, bvec, opdivfreevec>);}

        template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
        void Setbsigmahat()
        { bsigmahat = new FunctionCoefficient(bsigmahatTemplate<S, bvec, opdivfreevec>);}

        template<double (*S)(const Vector & xt), void (*bvec)(const Vector&, Vector& ),
                 void(*opdivfreevec)(const Vector & x, Vector & vec)> \
        void Setsigmahat()
        { sigmahat = new VectorFunctionCoefficient(dim, sigmahatTemplate<S, bvec, opdivfreevec>);}

        template<double (*S)(const Vector & xt), void (*bvec)(const Vector&, Vector& ),
                 void(*opdivfreevec)(const Vector & x, Vector & vec)> \
        void Setminsigmahat()
        { minsigmahat = new VectorFunctionCoefficient(dim, minsigmahatTemplate<S, bvec, opdivfreevec>);}

        template<double (*S)(const Vector & xt), void (*bvec)(const Vector&, Vector& )>
        void SetbSnonhomoVec()
        { bSnonhomo = new VectorFunctionCoefficient(dim, bSnonhomoTemplate<S, bvec>);}

};

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),  \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt), \
         void(*divfreevec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
void Transport_test_divfree::SetTestCoeffs ()
{
    SetSFun(S);
    SetSNonhomo<S>();
    SetScalarfFun<S, dSdt, Sgradxvec, bvec, divbfunc>();
    SetminbVec<bvec>();
    SetbVec(bvec);
    SetbSnonhomoVec<S, bvec>();
    SetbfVec<S, dSdt, Sgradxvec, bvec, divbfunc>();
    SetbdivsigmaVec<S, dSdt, Sgradxvec, bvec, divbfunc>();
    SetHdivVec<S,bvec>();
    SetKtildaMat<bvec>();
    SetScalarBtB<bvec>();
    SetminbTbSnonhomo<S, bvec>();
    SetSigmaNonhomoVec<S,bvec>();
    SetdivSigma<S, dSdt, Sgradxvec, bvec, divbfunc>();
    SetDivfreePart(divfreevec);
    SetOpDivfreePart(opdivfreevec);
    SetminKsigmahat<S, bvec, opdivfreevec>();
    Setbsigmahat<S, bvec, opdivfreevec>();
    Setsigmahat<S, bvec, opdivfreevec>();
    Setminsigmahat<S, bvec, opdivfreevec>();
    SetBBtMat<bvec>();
    return;
}


bool Transport_test_divfree::CheckTestConfig()
{
    if (dim == 4 || dim == 3)
    {
        if ( numsol == 0 && dim >= 3 )
            return true;
        if ( numsol == 1 && dim == 3 )
            return true;
        if ( numsol == 2 && dim == 3 )
            return true;
        if ( numsol == 4 && dim == 3 )
            return true;
        if ( numsol == -3 && dim == 3 )
            return true;
        if ( numsol == -4 && dim == 4 )
            return true;
        return false;
    }
    else
        return false;

}

Transport_test_divfree::Transport_test_divfree (int Dim, int NumSol, int NumCurl)
{
    dim = Dim;
    numsol = NumSol;
    numcurl = NumCurl;

    if ( CheckTestConfig() == false )
        std::cout << "Inconsistent dim = " << dim << " and numsol = " << numsol <<  std::endl << std::flush;
    else
    {
        if (numsol == 0)
        {
            if (dim == 3)
            {
                if (numcurl == 1)
                    SetTestCoeffs<&zero_ex, &zero_ex, &zerovecx_ex, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
                else if (numcurl == 2)
                    SetTestCoeffs<&zero_ex, &zero_ex, &zerovecx_ex, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
                else
                    SetTestCoeffs<&zero_ex, &zero_ex, &zerovecx_ex, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &zerovec_ex, &zerovec_ex>();
            }
            if (dim > 3)
            {
                if (numcurl == 1)
                    SetTestCoeffs<&zero_ex, &zero_ex, &zerovecx_ex, &bFunCube3D_ex, &bFunCube3Ddiv_ex, &DivmatFun4D_ex, &DivmatDivmatFun4D_ex>();
                else
                    SetTestCoeffs<&zero_ex, &zero_ex, &zerovecx_ex, &bFunCube3D_ex, &bFunCube3Ddiv_ex, &zerovec_ex, &zerovec_ex>();
            }
        }
        if (numsol == -3)
        {
            if (numcurl == 1)
                SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &zerovec_ex, &zerovec_ex>();
        }
        if (numsol == -4)
        {
            if (numcurl == 1)
                SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex, &zerovec_ex, &zerovec_ex>();
        }
        if (numsol == 1)
        {
            if (numcurl == 1)
                SetTestCoeffs<&uFun1_ex, &uFun1_ex_dt, &uFun1_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun1_ex, &uFun1_ex_dt, &uFun1_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun1_ex, &uFun1_ex_dt, &uFun1_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &zerovec_ex, &zerovec_ex>();
        }
        if (numsol == 2)
        {
            //std::cout << "The domain must be a cylinder over a square" << std::endl << std::flush;
            if (numcurl == 1)
                SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &zerovec_ex, &zerovec_ex>();
        }
        if (numsol == 4)
        {
            //std::cout << "The domain must be a cylinder over a square" << std::endl << std::flush;
            if (numcurl == 1)
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &zerovec_ex, &zerovec_ex>();
        }
    } // end of setting test coefficients in correct case
}
*/

class Heat_test_divfree
{
protected:
    int dim;
    int numsol;
    int numcurl;
    bool testisgood;

public:
    FunctionCoefficient * scalarS;                // S
    FunctionCoefficient * scalarSnonhomo;         // S(t=0)
    FunctionCoefficient * scalarf;                // = dS/dt - laplace S + laplace S(t=0) - what is used for solving
    FunctionCoefficient * scalardivsigma;         // = dS/dt - laplace S                  - what is used for computing error
    VectorFunctionCoefficient * sigma;
    VectorFunctionCoefficient * sigma_nonhomo;    // to incorporate inhomogeneous boundary conditions, stores (conv *S0, S0) with S(t=0) = S0
    VectorFunctionCoefficient * sigmahat;         // sigma_hat = sigma_exact - op divfreepart (curl hcurlpart in 3D)
    VectorFunctionCoefficient * divfreepart;      // additional part added for testing div-free solver
    VectorFunctionCoefficient * opdivfreepart;    // curl of the additional part which is added to sigma_exact for testing div-free solver
    VectorFunctionCoefficient * minsigmahat;      // -sigma_hat
public:
    Heat_test_divfree (int Dim, int NumSol, int NumCurl);

    int GetDim() {return dim;}
    int GetNumSol() {return numsol;}
    int GetNumCurl() {return numcurl;}
    int CheckIfTestIsGood() {return testisgood;}
    void SetDim(int Dim) { dim = Dim;}
    void SetNumSol(int NumSol) { numsol = NumSol;}
    void SetNumCurl(int NumCurl) { numcurl = NumCurl;}
    bool CheckTestConfig();

    ~Heat_test_divfree () {}
private:
    void SetScalarFun( double (*f)(const Vector & xt))
    { scalarS = new FunctionCoefficient(f);}

    template<double (*S)(const Vector & xt)> \
    void SetScalarSnonhomo()
    { scalarSnonhomo = new FunctionCoefficient(SnonhomoTemplate<S>);}

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    void SetRhandFun()
    { scalarf = new FunctionCoefficient(rhsideTemplate<S, dSdt, Slaplace>);}

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    void SetDivSigma()
    { scalardivsigma = new FunctionCoefficient(divsigmaTemplate<S, dSdt, Slaplace>);}


    template<double (*f1)(const Vector & xt), void(*f2)(const Vector & x, Vector & vec)> \
    void SetHdivFun()
    {
        sigma = new VectorFunctionCoefficient(dim, sigmaTemplate<f1,f2>);
    }

    template<double (*f1)(const Vector & xt), void(*f2)(const Vector & x, Vector & vec)> \
    void SetInitCondVec()
    {
        sigma_nonhomo = new VectorFunctionCoefficient(dim, sigmaNonHomoTemplate<f1,f2>);
    }

    void SetDivfreePart( void(*divfreevec)(const Vector & x, Vector & vec))
    { divfreepart = new VectorFunctionCoefficient(dim, divfreevec);}

    void SetOpDivfreePart( void(*opdivfreevec)(const Vector & x, Vector & vec))
    { opdivfreepart = new VectorFunctionCoefficient(dim, opdivfreevec);}

    template<double (*S)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),
             void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void Setsigmahat()
    { sigmahat = new VectorFunctionCoefficient(dim, sigmahatTemplate<S, Sgradxvec, opdivfreevec>);}

    template<double (*S)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),
             void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void Setminsigmahat()
    { minsigmahat = new VectorFunctionCoefficient(dim, minsigmahatTemplate<S, Sgradxvec, opdivfreevec>);}


    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt),
             double (*Slaplace)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
             void(*divfreevec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void SetTestCoeffs ( );
};


template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt),
         double (*Slaplace)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*divfreevec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
void Heat_test_divfree::SetTestCoeffs ()
{
    SetScalarFun(S);
    SetScalarSnonhomo<S>();
    SetRhandFun<S, dSdt, Slaplace>();
    SetHdivFun<S,Sgradxvec>();
    SetInitCondVec<S,Sgradxvec>();
    SetDivSigma<S, dSdt, Slaplace>();
    SetDivfreePart(divfreevec);
    SetOpDivfreePart(opdivfreevec);
    Setsigmahat<S, Sgradxvec, opdivfreevec>();
    Setminsigmahat<S, Sgradxvec, opdivfreevec>();
    return;
}


bool Heat_test_divfree::CheckTestConfig()
{
    if (dim == 4 || dim == 3)
    {
        if (numsol == 0 || numsol == 1)
            return true;
        if (numsol == 2 && dim == 4)
            return true;
        if (numsol == 3 && dim == 3)
            return true;
        if (numsol == -34 && (dim == 3 || dim == 4))
            return true;
        return false;
    }
    else
        return false;

}

Heat_test_divfree::Heat_test_divfree (int Dim, int NumSol, int NumCurl)
{
    dim = Dim;
    numsol = NumSol;
    numcurl = NumCurl;

    if ( CheckTestConfig() == false )
    {
        std::cerr << "Inconsistent dim and numsol \n" << std::flush;
        testisgood = false;
    }
    else
    {
        if (numsol == -34)
        {
            SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_laplace, &uFunTest_ex_gradx, &zerovec_ex, &zerovec_ex>();
        }
        if (numsol == 0)
        {
            //std::cout << "The domain should be either a unit rectangle or cube" << std::endl << std::flush;
            if (numcurl == 1)
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_laplace, &uFun_ex_gradx, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_laplace, &uFun_ex_gradx, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_laplace, &uFun_ex_gradx, &zerovec_ex, &zerovec_ex>();
        }
        if (numsol == 1)
        {
            //std::cout << "The domain should be either a unit rectangle or cube" << std::endl << std::flush;
            if (numcurl == 1)
                SetTestCoeffs<&uFun1_ex, &uFun1_ex_dt, &uFun1_ex_laplace, &uFun1_ex_gradx, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun1_ex, &uFun1_ex_dt, &uFun1_ex_laplace, &uFun1_ex_gradx, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun1_ex, &uFun1_ex_dt, &uFun1_ex_laplace, &uFun1_ex_gradx, &zerovec_ex, &zerovec_ex>();
        }
        if (numsol == 2)
        {
            if (numcurl == 1)
                SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_laplace, &uFun2_ex_gradx, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_laplace, &uFun2_ex_gradx, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_laplace, &uFun2_ex_gradx, &zerovec_ex, &zerovec_ex>();
        }
        if (numsol == 3)
        {
            if (numcurl == 1)
                SetTestCoeffs<&uFun3_ex, &uFun3_ex_dt, &uFun3_ex_laplace, &uFun3_ex_gradx, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun3_ex, &uFun3_ex_dt, &uFun3_ex_laplace, &uFun3_ex_gradx, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun3_ex, &uFun3_ex_dt, &uFun3_ex_laplace, &uFun3_ex_gradx, &zerovec_ex, &zerovec_ex>();
        }
        testisgood = true;
    }
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
    int numsol          = 3;
    int numcurl         = 0;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 2;

    bool withDiv = false;
    bool with_multilevel = false;
    //bool withS = true;
    //bool blockedversion = true;

    // solver options
    int prec_option = 0;        // defines whether to use preconditioner or not, and which one
    bool prec_is_MG;

    //const char *mesh_file = "../data/cube_3d_fine.mesh";
    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    //const char *mesh_file = "../data/cube4d_low.MFEM";
    //const char *mesh_file = "../data/cube4d_96.MFEM";
    //const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";

    int feorder         = 0;

    kappa = freq * M_PI;

    if (verbose)
        cout << "Solving CFOSLS Transport equation with MFEM & hypre, div-free approach \n";

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
    args.AddOption(&with_multilevel, "-ml", "--multilvl", "-no-ml",
                   "--no-multilvl",
                   "Enable or disable multilevel algorithm for finding a particular solution.");

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

    mesh_file = "../data/cube_3d_moderate.mesh";
    /*
    if (verbose)
        std::cout << "Running tests for the paper: \n";

    if (nDimensions == 3)
    {
        numsol = -34;
        mesh_file = "../data/cube_3d_moderate.mesh";
    }
    else // 4D case
    {
        numsol = -34;
        mesh_file = "../data/cube4d_96.MFEM";
    }
    */

    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";

    if (verbose)
        cout << "Number of mpi processes: " << num_procs << endl << flush;

    bool with_prec;

    switch (prec_option)
    {
    case 1: // smth simple like AMS
        with_prec = true;
        prec_is_MG = false;
        break;
    case 2: // MG
        with_prec = true;
        prec_is_MG = true;
        break;
    default: // no preconditioner (default)
        with_prec = false;
        prec_is_MG = false;
        break;
    }

    if (verbose)
    {
        cout << "with_prec = " << with_prec << endl;
        cout << "prec_is_MG = " << prec_is_MG << endl;
        cout << flush;
    }

    StopWatch chrono;

    //DEFAULTED LINEAR SOLVER OPTIONS
    int max_num_iter = 150000;
    double rtol = 1e-12;//1e-7;//1e-9;
    double atol = 1e-14;//1e-9;//1e-12;

    Mesh *mesh = NULL;

    shared_ptr<ParMesh> pmesh;

    if (nDimensions == 3 || nDimensions == 4)
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
    else //if nDimensions is not 3 or 4
    {
        if (verbose)
            cerr << "Case nDimensions = " << nDimensions << " is not supported \n" << std::flush;
        MPI_Finalize();
        return -1;
    }

    if (mesh) // if only serial mesh was generated previously, parallel mesh is initialized here
    {
        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        if (verbose)
            cout << "Creating parmesh(" << nDimensions <<
                    "d) from the serial mesh (" << nDimensions << "d)" << endl << flush;
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }

    int dim = nDimensions;
    //int sdim = nDimensions; // used in 4D case

    FiniteElementCollection *hdiv_coll;
    ParFiniteElementSpace *R_space;
    FiniteElementCollection *l2_coll;
    ParFiniteElementSpace *W_space;

    if (dim == 4)
        hdiv_coll = new RT0_4DFECollection;
    else
        hdiv_coll = new RT_FECollection(feorder, dim);

    R_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);

    if (withDiv)
    {
        l2_coll = new L2_FECollection(feorder, nDimensions);
        W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);
    }

    Array<int> ess_bdrU(pmesh->bdr_attributes.Max());
    ess_bdrU = 0;

    // data for multilevel algorithm
    const int ref_levels = par_ref_levels;
    DivPart divp;
    Vector sigmahat_pau;
    HypreParMatrix* d_td_coarse_R;
    HypreParMatrix* d_td_coarse_W;
    Array< SparseMatrix*> P_W(ref_levels);
    Array< SparseMatrix*> P_R(ref_levels);
    Array< SparseMatrix*> Element_dofs_R(ref_levels);
    Array< SparseMatrix*> Element_dofs_W(ref_levels);
    Array<int> ess_dof_coarsestlvl_list;

    if (with_multilevel)
    {
        if (!withDiv && verbose)
        {
            std::cout << "The multilevel code cannot be used without withDiv flag \n";
            MPI_Finalize();
            return 0;
        }
        else
            std::cout << "Using multilevel code for finding the particular solution \n";

        ParFiniteElementSpace *coarseR_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);
        ParFiniteElementSpace *coarseW_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

        // Input to the algorithm::

       // Array< int * > Sol_sig_level(ref_levels);

        const SparseMatrix* P_W_local;
        const SparseMatrix* P_R_local;

        // Dofs_TrueDofs at each coarse space:
        d_td_coarse_R = coarseR_space->Dof_TrueDof_Matrix();
        d_td_coarse_W = coarseW_space->Dof_TrueDof_Matrix();

        /*
        HypreParMatrix * temp_dof_TrueDof_R = R_space->Dof_TrueDof_Matrix();
        HypreParMatrix * temp_dof_TrueDof_W = W_space->Dof_TrueDof_Matrix();
        temp_dof_TrueDof_R->SetOwnerFlags(-1, -1, -1);
        temp_dof_TrueDof_W->SetOwnerFlags(-1, -1, -1);
        //LoseData();
        //temp_dof_TrueDof_W->LoseData();
        d_td_coarse_R = new HypreParMatrix(R_space->Dof_TrueDof_Matrix()->StealData());
        d_td_coarse_W = new HypreParMatrix(W_space->Dof_TrueDof_Matrix()->StealData());

        d_td_coarse_R->SetOwnerFlags(3, 3, 1);
        d_td_coarse_W->SetOwnerFlags(3, 3, 1);

        //d_td_coarse_R->MakeRef(*R_space->Dof_TrueDof_Matrix());
        //d_td_coarse_W->MakeRef(*W_space->Dof_TrueDof_Matrix());

        //R_space->Lose_Dof_TrueDof_Matrix();
        //W_space->Lose_Dof_TrueDof_Matrix();
        */

        for (int l = 0; l < ref_levels+1; l++)
        {
            if (l > 0)
            {
                if (l == 1)
                {
                    ess_bdrU=0;
                    //ess_bdrU=1;
                    //ess_bdrU[pmesh->bdr_attributes.Max() - 1] = 0;
                    R_space->GetEssentialVDofs(ess_bdrU, ess_dof_coarsestlvl_list);
                    //ess_dof_list.Print();
                }
                pmesh->UniformRefinement();

                P_W_local = ((const SparseMatrix *)W_space->GetUpdateOperator());
                P_R_local = ((const SparseMatrix *)R_space->GetUpdateOperator());

                SparseMatrix* R_Element_to_dofs1 = new SparseMatrix();
                SparseMatrix* W_Element_to_dofs1 = new SparseMatrix();

                divp.Elem2Dofs(*R_space, *R_Element_to_dofs1);
                divp.Elem2Dofs(*W_space, *W_Element_to_dofs1);

                P_W[ref_levels -l] = new SparseMatrix ( *P_W_local);
                P_R[ref_levels -l] = new SparseMatrix ( *P_R_local);

                Element_dofs_R[ref_levels - l] = R_Element_to_dofs1;
                Element_dofs_W[ref_levels - l] = W_Element_to_dofs1;
            }
        }

    }
    else
    {
        for (int l = 0; l < par_ref_levels; l++)
        {
            pmesh->UniformRefinement();
            if (withDiv)
                 W_space->Update();
            R_space->Update();
        }
    }

    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << "\n";

    Heat_test_divfree Mytest(nDimensions, numsol, numcurl);

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.

    shared_ptr<mfem::HypreParMatrix> A;
    ParBilinearForm *Ablock;
    ParLinearForm *ffform;

    FiniteElementCollection *hdivfree_coll;
    ParFiniteElementSpace *C_space;

    if (dim == 3)
    {
        hdivfree_coll = new ND_FECollection(feorder + 1, nDimensions);
        C_space = new ParFiniteElementSpace(pmesh.get(), hdivfree_coll);
    }
    else // dim == 4
    {
        if (verbose)
            std::cout << "4D case is not implemented yet \n";
        MPI_Finalize();
        return 0;
    } // end of initialization of div-free f.e. space in 4D

    FiniteElementCollection *h1_coll;
    ParFiniteElementSpace *H_space;
    h1_coll = new H1_FECollection(feorder+1, nDimensions);
    H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);

    int numblocks = 2;

    Array<int> block_offsets(numblocks + 1); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = C_space->GetVSize();
    block_offsets[2] = H_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(numblocks + 1); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = C_space->TrueVSize();
    block_trueOffsets[2] = H_space->TrueVSize();
    block_trueOffsets.PartialSum();

    HYPRE_Int dimC = C_space->GlobalTrueVSize();
    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimH = H_space->GlobalTrueVSize();
    if (verbose)
    {
       std::cout << "***********************************************************\n";
       std::cout << "dim(C) = " << dimC << "\n";
       std::cout << "dim(H) = " << dimH << ", ";
       std::cout << "dim(C+H) = " << dimC + dimH << "\n";
       if (withDiv)
           std::cout << "dim(R) = " << dimR << "\n";
       std::cout << "***********************************************************\n";
    }

    BlockVector xblks(block_offsets), rhsblks(block_offsets);
    BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
    xblks = 0.0;
    rhsblks = 0.0;
    trueX = 0.0;
    trueRhs = 0.0;

    //VectorFunctionCoefficient f(dim, f_exact);
    //VectorFunctionCoefficient vone(dim, vone_exact);
    //VectorFunctionCoefficient vminusone(dim, vminusone_exact);
    //VectorFunctionCoefficient E(dim, E_exact);
    //VectorFunctionCoefficient curlE(dim, curlE_exact);

    //----------------------------------------------------------
    // Setting boundary conditions.
    //----------------------------------------------------------

    if (!with_multilevel)
    {
        //ess_bdrU = 0;
        //ess_bdrU[0] = 1;
        //ess_bdrU[1] = 1;
        //ess_bdrU = 1;
        //ess_bdrU[pmesh->bdr_attributes.Max() - 1] = 0;
    }
    //Array<int> ess_tdof_listU;
    //C_space->GetEssentialTrueDofs(ess_bdrU, ess_tdof_listU);

    Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
    ess_bdrS = 0;
    //ess_bdrS[0] = 1; // t = 0
    ess_bdrS = 1;
    ess_bdrS[pmesh->bdr_attributes.Max() - 1] = 0;
    //Array<int> ess_tdof_listSж
    //H_space->GetEssentialTrueDofs(ess_bdrS, ess_tdof_listS);

    ParGridFunction * Sigmahat = new ParGridFunction(R_space);
    if (withDiv)
    {
        if (verbose)
            std::cout << "Assembling linear system for finding sigmahat \n";

        if (with_multilevel)
        {
            int ref_levels = par_ref_levels;

            if (verbose)
                std::cout << "Running Paulina's code \n";

            // Setting boundary conditions if any:

            // Define the coefficients, analytical solution, and rhs of the PDE.
            ConstantCoefficient k(1.0);
            //FunctionCoefficient fcoeff(fFun);


            ParBilinearForm *mVarf(new ParBilinearForm(R_space));
            mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
            mVarf->Assemble();
            mVarf->Finalize();
            SparseMatrix &M_fine(mVarf->SpMat());

            ParMixedBilinearForm *bVarf(new ParMixedBilinearForm(R_space, W_space));
            bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
            bVarf->Assemble();
            bVarf->Finalize();
            SparseMatrix &B_fine = bVarf->SpMat();

            SparseMatrix *M_local = &M_fine;


            SparseMatrix *B_local = &B_fine;

            //Right hand size
            Vector F_fine(P_W[0]->Height());
            Vector G_fine(P_R[0]->Height());


            ParLinearForm gform(R_space);
            gform.Update();
            gform.Assemble();

            ParLinearForm fform(W_space);
            fform.Update();
            fform.AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
            fform.Assemble();

            F_fine = fform;
            G_fine = .0;

            divp.div_part(ref_levels,
                     M_local, B_local,
                     G_fine,
                     F_fine,
                     P_W, P_R, P_W,
                     Element_dofs_R,
                     Element_dofs_W,
                     d_td_coarse_R,
                     d_td_coarse_W,
                     sigmahat_pau, ess_dof_coarsestlvl_list);


            Vector sth(F_fine.Size());

            bVarf->SpMat().Mult(sigmahat_pau, sth);
            sth -= F_fine;

            cout<< "final check " << sth.Norml2() <<endl;

            *Sigmahat = sigmahat_pau;
        }
        else
        {
            ParGridFunction *sigma_exact;
            ParLinearForm *gform(new ParLinearForm);
            ParMixedBilinearForm *Bblock;
            HypreParMatrix *Bdiv, *BdivT;
            HypreParMatrix *BBT;
            HypreParVector *Rhs;

            sigma_exact = new ParGridFunction(R_space);
            sigma_exact->ProjectCoefficient(*(Mytest.sigma));

            gform = new ParLinearForm(W_space);
            gform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
            gform->Assemble();

            Bblock = new ParMixedBilinearForm(R_space, W_space);
            Bblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
            Bblock->Assemble();
            Bblock->EliminateTrialDofs(ess_bdrU, *sigma_exact, *gform);

            Bblock->Finalize();
            Bdiv = Bblock->ParallelAssemble();
            BdivT = Bdiv->Transpose();
            BBT = ParMult(Bdiv, BdivT);
            Rhs = gform->ParallelAssemble();

            HypreBoomerAMG * invBBT = new HypreBoomerAMG(*BBT);
            invBBT->SetPrintLevel(0);

            mfem::CGSolver solver(comm);
            solver.SetPrintLevel(1);
            solver.SetMaxIter(70000);
            solver.SetRelTol(1.0e-16);
            solver.SetAbsTol(1.0e-16);
            solver.SetPreconditioner(*invBBT);
            solver.SetOperator(*BBT);

            Vector * Temphat = new Vector(W_space->TrueVSize());
            solver.Mult(*Rhs, *Temphat);

            Vector * Temp = new Vector(R_space->TrueVSize());
            BdivT->Mult(*Temphat, *Temp);

            Sigmahat->Distribute(*Temp);
            //Sigmahat->SetFromTrueDofs(*Temp);
        }

    }
    else // solving a div-free system with some analytical solution for the div-free part
    {
        Sigmahat->ProjectCoefficient(*(Mytest.sigmahat));
    }
    // in either way now Sigmahat is a function from H(div) s.t. div Sigmahat = div sigma = f

    MFEM_ASSERT(dim == 3, "For now only 3D case is considered \n");
    if (nDimensions == 4)
    {
        if (verbose)
            std::cout << "4D case is not implemented - not a curl problem should be solved there! \n";
        MPI_Finalize();
        return 0;
    }

    // the div-free part
    ParGridFunction *u_exact = new ParGridFunction(C_space);
    u_exact->ProjectCoefficient(*(Mytest.divfreepart));

    ParGridFunction *S_exact = new ParGridFunction(H_space);
    S_exact->ProjectCoefficient(*(Mytest.scalarS));

    if (withDiv)
        xblks.GetBlock(0) = 0.0;
    else
        xblks.GetBlock(0) = *u_exact;
    xblks.GetBlock(1) = *S_exact;


    ffform = new ParLinearForm(C_space);
    /*
    if (!withDiv)
    {
        ffform->Update(C_space, rhsblks.GetBlock(0), 0);
        ffform->AddDomainIntegrator(new VectorcurlDomainLFIntegrator(*(Mytest.minsigmahat)));
        ffform->Assemble();
    }
    else // if withDiv = true
    */
    {
        ParMixedBilinearForm * Tblock = new ParMixedBilinearForm(R_space, C_space);
        // integrates (phi, curl psi)
        Tblock->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator);
        Tblock->Assemble();
        Tblock->EliminateTestDofs(ess_bdrU);
        Tblock->Finalize();
        // since we need (- sigmahat, curl psi), we change the sign here
        *Sigmahat *= -1.0;
        Tblock->Mult(*Sigmahat, *ffform);
        *Sigmahat *= -1.0;
    }

    ParLinearForm *qform(new ParLinearForm);
    /*
    if (!withDiv)
    {
        qform->Update(H_space, rhsblks.GetBlock(1), 0);
        //qform->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bdivsigma));
        //qform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.bsigmahat));
        qform->AddDomainIntegrator(new HeatSigmaLFIntegrator(*Mytest.sigmahat));
        qform->Assemble();
    }
    else
    */
    {
        qform->Update(H_space, rhsblks.GetBlock(1), 0);

        //qform->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bdivsigma));
        //qform->Assemble();

        ParMixedBilinearForm * LTblock = new ParMixedBilinearForm(H_space, R_space);
        // integrates (phi, (-q, grad_x q)^T)
        LTblock->AddDomainIntegrator(new PAUVectorFEMassIntegrator);
        LTblock->Assemble();
        LTblock->EliminateTestDofs(ess_bdrS);
        LTblock->Finalize();
        // we need to change the sign for the rh side of our system
        *Sigmahat *= -1.0;
        LTblock->MultTranspose(*Sigmahat, *qform);
        *Sigmahat *= -1.0;
    }

    Ablock = new ParBilinearForm(C_space);
    Coefficient *one = new ConstantCoefficient(1.0);
    // integrates (curl phi, curl psi)
    Ablock->AddDomainIntegrator(new CurlCurlIntegrator(*one));
    Ablock->Assemble();
    Ablock->EliminateEssentialBC(ess_bdrU,xblks.GetBlock(0),*ffform);
    Ablock->Finalize();
    HypreParMatrix * tempA = Ablock->ParallelAssemble();
    A = make_shared<HypreParMatrix>(*tempA);

    ParBilinearForm *Cblock;
    HypreParMatrix *C;

    Cblock = new ParBilinearForm(H_space);
    // integrates ((-q, grad_x q)^T, (-p, grad_x p)^T)
    Cblock->AddDomainIntegrator(new PAUVectorFEMassIntegrator2);
    //Cblock->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));
    //Cblock->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.bbT));
    Cblock->Assemble();
    Cblock->EliminateEssentialBC(ess_bdrS, xblks.GetBlock(1),*qform);
    Cblock->Finalize();
    C = Cblock->ParallelAssemble();

    ParMixedBilinearForm *CHblock;
    HypreParMatrix *CH, *CHT;

    CHblock = new ParMixedBilinearForm(C_space, H_space);
    // integrates (curl phi, (-p, grad_x p)^T)
    CHblock->AddDomainIntegrator(new HeatVectorFECurlSigmaIntegrator);
    //CHblock->AddDomainIntegrator(new VectorFECurlVQIntegrator(*Mytest.minb));
    CHblock->Assemble();
    CHblock->EliminateTestDofs(ess_bdrS);
    CHblock->EliminateTrialDofs(ess_bdrU, xblks.GetBlock(0), *qform);
    CHblock->Finalize();

    CH = CHblock->ParallelAssemble();
    CHT = CH->Transpose();

    ffform->ParallelAssemble(trueRhs.GetBlock(0));
    qform->ParallelAssemble(trueRhs.GetBlock(1));

    BlockOperator *MainOp = new BlockOperator(block_trueOffsets);

    MainOp->SetBlock(0,0, A.get());
    MainOp->SetBlock(0,1, CHT);
    MainOp->SetBlock(1,0, CH);
    MainOp->SetBlock(1,1, C);

    if (verbose)
        cout << "Discretized problem is assembled" << endl << flush;

    chrono.Clear();
    chrono.Start();

    Solver *prec;
    if (with_prec)
    {
        if(dim<=3)
        {
            if (prec_is_MG)
            {
                if (verbose)
                    cout << "MG prec is not implemented" << endl;
                MPI_Finalize();
                return 0;

                //int formcurl = 1; // for H(curl)
                //prec = new MG3dPrec(&Amat, nlevels, coarsenfactor, pmesh.get(), formcurl, feorder, C_space, ess_tdof_listU, verbose);
            }
            else // prec is AMS-like for the div-free part (block-diagonal for the system with boomerAMG for S)
            {
                prec = new BlockDiagonalPreconditioner(block_trueOffsets);
                Operator * precU = new HypreAMS(*A, C_space);
                ((HypreAMS*)precU)->SetSingularProblem();
                Operator * precS = new HypreBoomerAMG(*C);
                ((HypreBoomerAMG*)precS)->SetPrintLevel(0);

                ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, precU);
                ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(1, precS);
            }
        }
        else // if(dim==4)
        {
            if (prec_is_MG)
            {
                if (verbose)
                    cout << "MG prec is not implemented in 4D" << endl;
                MPI_Finalize();
                return 0;
            }
        }

        if (verbose)
            cout << "Preconditioner is ready" << endl << flush;
    }
    else
        if (verbose)
            cout << "Using no preconditioner" << endl << flush;

    IterativeSolver * solver;
    solver = new CGSolver(comm);
    if (verbose)
        cout << "Linear solver: CG" << endl << flush;

    solver->SetAbsTol(atol);
    solver->SetRelTol(rtol);
    solver->SetMaxIter(max_num_iter);
    solver->SetOperator(*MainOp);

    if (with_prec)
        solver->SetPreconditioner(*prec);
    solver->SetPrintLevel(0);
    trueX = 0.0;
    solver->Mult(trueRhs, trueX);
    chrono.Stop();

    if (verbose)
    {
       if (solver->GetConverged())
          std::cout << "Linear solver converged in " << solver->GetNumIterations()
                    << " iterations with a residual norm of " << solver->GetFinalNorm() << ".\n";
       else
          std::cout << "Linear solver did not converge in " << solver->GetNumIterations()
                    << " iterations. Residual norm is " << solver->GetFinalNorm() << ".\n";
       std::cout << "Linear solver took " << chrono.RealTime() << "s. \n";
    }

    ParGridFunction * u = new ParGridFunction(C_space);
    ParGridFunction * S;

    u->Distribute(&(trueX.GetBlock(0)));
    S = new ParGridFunction(H_space);
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

    double err_u = u->ComputeL2Error(*(Mytest.divfreepart), irs);
    double norm_u = ComputeGlobalLpNorm(2, *(Mytest.divfreepart), *pmesh, irs);

    if (verbose && !withDiv)
    {
        if ( norm_u > MYZEROTOL )
        {
            //std::cout << "norm_u = " << norm_u << "\n";
            cout << "|| u - u_ex || / || u_ex || = " << err_u / norm_u << endl;
        }
        else
            cout << "|| u || = " << err_u << " (u_ex = 0)" << endl;
    }

    // if we are solving without finding a particular solution, just a simple system in div-free space
    // then we replace our Sigmahat (which will be a prt of final sigma) by the projection of the artificial
    // sigmahat related to the div-free system
    if (!withDiv)
        Sigmahat->ProjectCoefficient(*(Mytest.sigmahat));

    ParGridFunction * divfreepart = new ParGridFunction(R_space);
    DiscreteLinearOperator Curl_h(C_space, R_space);
    Curl_h.AddDomainInterpolator(new CurlInterpolator());
    Curl_h.Assemble();
    Curl_h.Mult(*u, *divfreepart); // if replaced by u_exact, makes the error look nicer

    ParGridFunction * divfreepart_exact = new ParGridFunction(R_space);
    divfreepart_exact->ProjectCoefficient(*(Mytest.opdivfreepart));

    double err_divfreepart = divfreepart->ComputeL2Error(*(Mytest.opdivfreepart), irs);
    double norm_divfreepart = ComputeGlobalLpNorm(2, *(Mytest.opdivfreepart), *pmesh, irs);

    if (verbose && !withDiv)
    {
        if ( norm_divfreepart > MYZEROTOL )
        {
            //cout << "|| divfreepart_ex || = " << norm_divfreepart << endl;
            cout << "|| curl_h u_h - divfreepart_ex || / || divfreepart_ex || = " << err_divfreepart / norm_divfreepart << endl;
        }
        else
            cout << "|| curl_h u_h || = " << err_divfreepart << " (divfreepart_ex = 0)" << endl;
    }

    ParGridFunction * sigma = new ParGridFunction(R_space);
    *sigma = *Sigmahat; // particular solution
    *sigma += *divfreepart;   // plus div-free guy
    ParGridFunction * sigma_exact = new ParGridFunction(R_space);
    sigma_exact->ProjectCoefficient(*(Mytest.sigma));

    double err_sigma = sigma->ComputeL2Error(*(Mytest.sigma), irs);
    double norm_sigma = ComputeGlobalLpNorm(2, *(Mytest.sigma), *pmesh, irs);

    if (verbose)
        cout << "sigma_h = sigma_hat + div-free part, div-free part = curl u_h \n";

    if (verbose)
    {
        if ( norm_sigma > MYZEROTOL )
            cout << "|| sigma_h - sigma_ex || / || sigma_ex || = " << err_sigma / norm_sigma << endl;
        else
            cout << "|| sigma || = " << err_sigma << " (sigma_ex = 0)" << endl;
    }

    /*
    double err_sigmahat = Sigmahat->ComputeL2Error(*(Mytest.sigma), irs);
    if (verbose && !withDiv)
        if ( norm_sigma > MYZEROTOL )
            cout << "|| sigma_hat - sigma_ex || / || sigma_ex || = " << err_sigmahat / norm_sigma << endl;
        else
            cout << "|| sigma_hat || = " << err_sigmahat << " (sigma_ex = 0)" << endl;
    */

    double norm_S;
    S_exact = new ParGridFunction(H_space);
    S_exact->ProjectCoefficient(*(Mytest.scalarS));

    double err_S = S->ComputeL2Error(*(Mytest.scalarS), irs);
    norm_S = ComputeGlobalLpNorm(2, *(Mytest.scalarS), *pmesh, irs);
    if (verbose)
    {
        if ( norm_S > MYZEROTOL )
            std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                     err_S / norm_S << "\n";
        else
            std::cout << "|| S_h || = " << err_S << " (S_ex = 0) \n";
    }

    if (verbose)
        cout << "Computing projection errors" << endl;

    //double projection_error_u = u_exact->ComputeL2Error(E, irs);
    double projection_error_u = u_exact->ComputeL2Error(*(Mytest.divfreepart), irs);

    if(verbose && !withDiv)
    {
        if ( norm_u > MYZEROTOL )
        {
            //std::cout << "Debug: || u_ex || = " << norm_u << "\n";
            //std::cout << "Debug: proj error = " << projection_error_u << "\n";
            cout << "|| u_ex - Pi_h u_ex || / || u_ex || = " << projection_error_u / norm_u << endl;
        }
        else
            cout << "|| Pi_h u_ex || = " << projection_error_u << " (u_ex = 0) \n ";
    }

    double projection_error_sigma = sigma_exact->ComputeL2Error(*(Mytest.sigma), irs);

    if(verbose)
    {
        if ( norm_sigma > MYZEROTOL )
        {
            cout << "|| sigma_ex - Pi_h sigma_ex || / || sigma_ex || = " << projection_error_sigma / norm_sigma << endl;
        }
        else
            cout << "|| Pi_h sigma_ex || = " << projection_error_sigma << " (sigma_ex = 0) \n ";
    }
    double projection_error_S = S_exact->ComputeL2Error(*(Mytest.scalarS), irs);

    if(verbose)
    {
       if ( norm_S > MYZEROTOL )
           cout << "|| S_ex - Pi_h S_ex || / || S_ex || = " << projection_error_S / norm_S << endl;
       else
           cout << "|| Pi_h S_ex ||  = " << projection_error_S << " (S_ex = 0) \n";
    }

    if (visualization && nDimensions < 4)
    {
       char vishost[] = "localhost";
       int  visport   = 19916;


       socketstream uex_sock(vishost, visport);
       uex_sock << "parallel " << num_procs << " " << myid << "\n";
       uex_sock.precision(8);
       uex_sock << "solution\n" << *pmesh << *u_exact << "window_title 'u_exact'"
              << endl;
       socketstream uh_sock(vishost, visport);
       uh_sock << "parallel " << num_procs << " " << myid << "\n";
       uh_sock.precision(8);
       uh_sock << "solution\n" << *pmesh << *u << "window_title 'u_h'"
              << endl;

       *u -= *u_exact;
       socketstream udiff_sock(vishost, visport);
       udiff_sock << "parallel " << num_procs << " " << myid << "\n";
       udiff_sock.precision(8);
       udiff_sock << "solution\n" << *pmesh << *u << "window_title 'u_h - u_exact'"
              << endl;


       socketstream divfreepartex_sock(vishost, visport);
       divfreepartex_sock << "parallel " << num_procs << " " << myid << "\n";
       divfreepartex_sock.precision(8);
       divfreepartex_sock << "solution\n" << *pmesh << *divfreepart_exact << "window_title 'curl u_exact'"
              << endl;

       socketstream divfreepart_sock(vishost, visport);
       divfreepart_sock << "parallel " << num_procs << " " << myid << "\n";
       divfreepart_sock.precision(8);
       divfreepart_sock << "solution\n" << *pmesh << *divfreepart << "window_title 'curl u_h'"
              << endl;

       *divfreepart -= *divfreepart_exact;
       socketstream divfreepartdiff_sock(vishost, visport);
       divfreepartdiff_sock << "parallel " << num_procs << " " << myid << "\n";
       divfreepartdiff_sock.precision(8);
       divfreepartdiff_sock << "solution\n" << *pmesh << *divfreepart << "window_title 'curl u_h - curl u_exact'"
              << endl;

       socketstream S_ex_sock(vishost, visport);
       S_ex_sock << "parallel " << num_procs << " " << myid << "\n";
       S_ex_sock.precision(8);
       S_ex_sock << "solution\n" << *pmesh << *S_exact << "window_title 'S_exact'"
              << endl;

       socketstream S_h_sock(vishost, visport);
       S_h_sock << "parallel " << num_procs << " " << myid << "\n";
       S_h_sock.precision(8);
       S_h_sock << "solution\n" << *pmesh << *S << "window_title 'S_h'"
              << endl;

       *S -= *S_exact;
       socketstream S_diff_sock(vishost, visport);
       S_diff_sock << "parallel " << num_procs << " " << myid << "\n";
       S_diff_sock.precision(8);
       S_diff_sock << "solution\n" << *pmesh << *S << "window_title 'S_h - S_exact'"
              << endl;

       MPI_Barrier(pmesh->GetComm());
    }

    // 17. Free the used memory.
    delete ffform;
    delete qform;

    delete Ablock;
    delete Cblock;
    delete CHblock;

    delete C_space;
    delete hdivfree_coll;
    delete R_space;
    delete hdiv_coll;
    delete H_space;
    delete h1_coll;

    MPI_Finalize();
    return 0;
}

void zerovecx_ex(const Vector& xt, Vector& zerovecx )
{
    zerovecx.SetSize(xt.Size() - 1);
    zerovecx = 0.0;
}

void zerovec_ex(const Vector& xt, Vector& vecvalue)
{
    vecvalue.SetSize(xt.Size());

    //vecvalue(0) = -y * (1 - t);
    //vecvalue(1) = x * (1 - t);
    //vecvalue(2) = 0;
    //vecvalue(0) = x * (1 - x);
    //vecvalue(1) = y * (1 - y);
    //vecvalue(2) = t * (1 - t);

    // Martin's function
    vecvalue = 0.0;

    return;
}

double zero_ex(const Vector& xt)
{
    return 0.0;
}

////////////////
void hcurlFun3D_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //vecvalue(0) = -y * (1 - t);
    //vecvalue(1) = x * (1 - t);
    //vecvalue(2) = 0;
    //vecvalue(0) = x * (1 - x);
    //vecvalue(1) = y * (1 - y);
    //vecvalue(2) = t * (1 - t);

    // Martin's function
    vecvalue(0) = sin(kappa * y);
    vecvalue(1) = sin(kappa * t);
    vecvalue(2) = sin(kappa * x);

    return;
}

void curlhcurlFun3D_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //vecvalue(0) = 0.0;
    //vecvalue(1) = 0.0;
    //vecvalue(2) = -2.0 * (1 - t);

    // Martin's function's curl
    vecvalue(0) = - kappa * cos(kappa * t);
    vecvalue(1) = - kappa * cos(kappa * x);
    vecvalue(2) = - kappa * cos(kappa * y);

    return;
}

////////////////
void hcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //
    vecvalue(0) = 100.0 * x * x * (1-x) * (1-x) * y * y * (1-y) * (1-y) * t * t * (1-t) * (1-t);
    vecvalue(1) = 0.0;
    vecvalue(2) = 0.0;

    return;
}

void curlhcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //
    vecvalue(0) = 0.0;
    vecvalue(1) = 100.0 * ( 2.0) * t * (1-t) * (1.-2.*t) * x * x * (1-x) * (1-x) * y * y * (1-y) * (1-y);
    vecvalue(2) = 100.0 * (-2.0) * y * (1-y) * (1.-2.*y) * x * x * (1-x) * (1-x) * t * t * (1-t) * (1-t);

    return;
}

template <double (*S)(const Vector&), void (*Sgradxvec)(const Vector&, Vector& )> \
void sigmaTemplate(const Vector& xt, Vector& sigma)
{
    sigma.SetSize(xt.Size());

    Vector gradS;
    Sgradxvec(xt,gradS);

    sigma(xt.Size()-1) = S(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = - gradS(i);

    return;
}

template <double (*S)(const Vector&), void (*Sgradxvec)(const Vector&, Vector& )> \
void sigmaNonHomoTemplate(const Vector& xt, Vector& sigma) // sigmaNonHomo = ( - grad u, u) for u = S(t=0)
{
    sigma.SetSize(xt.Size());

    Vector xteq0(xt.Size()); // xt with t = 0
    xteq0 = xt;
    xteq0(xteq0.Size()-1) = 0.0;

    Vector gradS;
    Sgradxvec(xteq0,gradS);

    sigma(xt.Size()-1) = S(xteq0);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = - gradS(i);

    return;
}


template <double (*S)(const Vector & xt), void (*Sgradxvec)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void sigmahatTemplate(const Vector& xt, Vector& sigmahatv)
{
    sigmahatv.SetSize(xt.Size());

    Vector sigma(xt.Size());
    sigmaTemplate<S, Sgradxvec>(xt, sigma);

    Vector opdivfree;
    opdivfreevec(xt, opdivfree);

    sigmahatv = 0.0;
    sigmahatv -= opdivfree;
    sigmahatv += sigma;
    return;
}

template <double (*S)(const Vector & xt), void (*Sgradxvec)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void minsigmahatTemplate(const Vector& xt, Vector& minsigmahatv)
{
    minsigmahatv.SetSize(xt.Size());
    sigmahatTemplate<S, Sgradxvec, opdivfreevec>(xt, minsigmahatv);
    minsigmahatv *= -1;

    return;
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt) > \
double rhsideTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return dSdt(xt) - Slaplace(xt) + Slaplace(xt0);
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt) > \
double divsigmaTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return dSdt(xt) - Slaplace(xt);
}

template<double (*S)(const Vector & xt)> \
    double SnonhomoTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return S(xt0);
}


double uFun_ex(const Vector & xt)
{
    const double PI = 3.141592653589793;
    double xi(xt(0));
    double yi(xt(1));
    double zi(0.0);
    double vi(0.0);

    if (xt.Size() == 3)
    {
        zi = xt(2);
        return sin(PI*xi)*sin(PI*yi)*zi;
    }
    if (xt.Size() == 4)
    {
        zi = xt(2);
        vi = xt(3);
        //cout << "sol for 4D" << endl;
        return sin(PI*xi)*sin(PI*yi)*sin(PI*zi)*vi;
    }

    return 0.0;
}


double uFun_ex_dt(const Vector & xt)
{
    const double PI = 3.141592653589793;
    double xi(xt(0));
    double yi(xt(1));
    double zi(0.0);

    if (xt.Size() == 3)
        return sin(PI*xi)*sin(PI*yi);
    if (xt.Size() == 4)
    {
        zi = xt(2);
        return sin(PI*xi)*sin(PI*yi)*sin(PI*zi);
    }

    return 0.0;
}

double uFun_ex_laplace(const Vector & xt)
{
    const double PI = 3.141592653589793;
    return (-(xt.Size()-1) * PI * PI) *uFun_ex(xt);
}

void uFun_ex_gradx(const Vector& xt, Vector& gradx )
{
    const double PI = 3.141592653589793;

    double x = xt(0);
    double y = xt(1);
    double z(0.0);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    if (xt.Size() == 3)
    {
        gradx(0) = t * PI * cos (PI * x) * sin (PI * y);
        gradx(1) = t * PI * sin (PI * x) * cos (PI * y);
    }
    if (xt.Size() == 4)
    {
        z = xt(2);
        gradx(0) = t * PI * cos (PI * x) * sin (PI * y) * sin (PI * z);
        gradx(1) = t * PI * sin (PI * x) * cos (PI * y) * sin (PI * z);
        gradx(2) = t * PI * sin (PI * x) * sin (PI * y) * cos (PI * z);
    }

}


double fFun(const Vector & x)
{
    const double PI = 3.141592653589793;
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    double vi(0.0);
    if (x.Size() == 3)
    {
     zi = x(2);
       return 2*PI*PI*sin(PI*xi)*sin(PI*yi)*zi+sin(PI*xi)*sin(PI*yi);
    }

    if (x.Size() == 4)
    {
     zi = x(2);
         vi = x(3);
         //cout << "rhand for 4D" << endl;
       return 3*PI*PI*sin(PI*xi)*sin(PI*yi)*sin(PI*zi)*vi + sin(PI*xi)*sin(PI*yi)*sin(PI*zi);
    }

    return 0.0;
}

void sigmaFun_ex(const Vector & x, Vector & u)
{
    const double PI = 3.141592653589793;
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    double vi(0.0);
    if (x.Size() == 3)
    {
        zi = x(2);
        u(0) = - PI * cos (PI * xi) * sin (PI * yi) * zi;
        u(1) = - PI * cos (PI * yi) * sin (PI * xi) * zi;
        u(2) = uFun_ex(x);
        return;
    }

    if (x.Size() == 4)
    {
        zi = x(2);
        vi = x(3);
        u(0) = - PI * cos (PI * xi) * sin (PI * yi) * sin(PI * zi) * vi;
        u(1) = - sin (PI * xi) * PI * cos (PI * yi) * sin(PI * zi) * vi;
        u(2) = - sin (PI * xi) * sin(PI * yi) * PI * cos (PI * zi) * vi;
        u(3) = uFun_ex(x);
        return;
    }

    if (x.Size() == 2)
    {
        u(0) =  exp(-PI*PI*yi)*PI*cos(PI*xi);
        u(1) = -sin(PI*xi)*exp(-1*PI*PI*yi);
        return;
    }

    return;
}



double uFun1_ex(const Vector & xt)
{
    double tmp = (xt.Size() == 4) ? sin(M_PI*xt(2)) : 1.0;
    return exp(-xt(xt.Size()-1))*sin(M_PI*xt(0))*sin(M_PI*xt(1))*tmp;
}

double uFun1_ex_dt(const Vector & xt)
{
    return - uFun1_ex(xt);
}

double uFun1_ex_laplace(const Vector & xt)
{
    return (- (xt.Size() - 1) * M_PI * M_PI ) * uFun1_ex(xt);
}

void uFun1_ex_gradx(const Vector& xt, Vector& gradx )
{
    const double PI = 3.141592653589793;

    double x = xt(0);
    double y = xt(1);
    double z(0.0);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    if (xt.Size() == 3)
    {
        gradx(0) = exp(-t) * PI * cos (PI * x) * sin (PI * y);
        gradx(1) = exp(-t) * PI * sin (PI * x) * cos (PI * y);
    }
    if (xt.Size() == 4)
    {
        z = xt(2);
        gradx(0) = exp(-t) * PI * cos (PI * x) * sin (PI * y) * sin (PI * z);
        gradx(1) = exp(-t) * PI * sin (PI * x) * cos (PI * y) * sin (PI * z);
        gradx(2) = exp(-t) * PI * sin (PI * x) * sin (PI * y) * cos (PI * z);
    }

}

double fFun1(const Vector & x)
{
    return ( (x.Size()-1)*M_PI*M_PI - 1. ) * uFun1_ex(x);
}

void sigmaFun1_ex(const Vector & x, Vector & sigma)
{
    sigma.SetSize(x.Size());
    sigma(0) = -M_PI*exp(-x(x.Size()-1))*cos(M_PI*x(0))*sin(M_PI*x(1));
    sigma(1) = -M_PI*exp(-x(x.Size()-1))*sin(M_PI*x(0))*cos(M_PI*x(1));
    if (x.Size() == 4)
    {
        sigma(0) *= sin(M_PI*x(2));
        sigma(1) *= sin(M_PI*x(2));
        sigma(2) = -M_PI*exp(-x(x.Size()-1))*sin(M_PI*x(0))
                *sin(M_PI*x(1))*cos(M_PI*x(2));
    }
    sigma(x.Size()-1) = uFun1_ex(x);

    return;
}

double uFun2_ex(const Vector & xt)
{
    if (xt.Size() != 4)
        cout << "Error, this is only 4-d solution" << endl;
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(3);

    return exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y) * (2 - z) * sin (M_PI * z);
}

double uFun2_ex_dt(const Vector & xt)
{
    return - uFun2_ex(xt);
}

double uFun2_ex_laplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(3);

    double res = 0.0;
    res += exp(-t) * (2.0 * M_PI * cos(M_PI * x) - x * M_PI * M_PI * sin (M_PI * x)) * (1 + y) * sin (M_PI * y) * (2 - z) * sin (M_PI * z);
    res += exp(-t) * x * sin (M_PI * x) * (2.0 * M_PI * cos(M_PI * y) - (1 + y) * M_PI * M_PI * sin(M_PI * y)) * (2 - z) * sin (M_PI * z);
    res += exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y) * (2.0 * (-1) * M_PI * cos(M_PI * z) - (2 - z) * M_PI * M_PI * sin(M_PI * z));
    return res;
}

void uFun2_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(3);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(-t) * (sin (M_PI * x) + x * M_PI * cos(M_PI * x)) * (1 + y) * sin (M_PI * y) * (2 - z) * sin (M_PI * z);
    gradx(1) = exp(-t) * x * sin (M_PI * x) * (sin (M_PI * y) + (1 + y) * M_PI * cos(M_PI * y)) * (2 - z) * sin (M_PI * z);
    gradx(2) = exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y) * (- sin (M_PI * z) + (2 - z) * M_PI * cos(M_PI * z));
}

double uFun3_ex(const Vector & xt)
{
    if (xt.Size() != 3)
        cout << "Error, this is only 3-d = 2-d + time solution" << endl;
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y);
}

double uFun3_ex_dt(const Vector & xt)
{
    return - uFun3_ex(xt);
}

double uFun3_ex_laplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    double res = 0.0;
    res += exp(-t) * (2.0 * M_PI * cos(M_PI * x) - x * M_PI * M_PI * sin (M_PI * x)) * (1 + y) * sin (M_PI * y);
    res += exp(-t) * x * sin (M_PI * x) * (2.0 * M_PI * cos(M_PI * y) - (1 + y) * M_PI * M_PI * sin(M_PI * y));
    return res;
}

void uFun3_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(-t) * (sin (M_PI * x) + x * M_PI * cos(M_PI * x)) * (1 + y) * sin (M_PI * y);
    gradx(1) = exp(-t) * x * sin (M_PI * x) * (sin (M_PI * y) + (1 + y) * M_PI * cos(M_PI * y));
}

