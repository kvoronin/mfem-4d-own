//
//                        MFEM CFOSLS Transport equation with multigrid (div-free part)
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

#include"cfosls_testsuite.hpp"

//#define BAD_TEST
//#define ONLY_DIVFREEPART
//#define K_IDENTITY

#define PAULINA_CODE

#define MYZEROTOL (1.0e-13)

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

#ifdef PAULINA_CODE
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

#endif

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
      //cout << "elvect = " << elvect << endl;
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
   int dimc;
   int vector_dof, scalar_dof;

   MFEM_ASSERT(trial_fe.GetMapType() == mfem::FiniteElement::H_CURL ||
               test_fe.GetMapType() == mfem::FiniteElement::H_CURL,
               "At least one of the finite elements must be in H(Curl)");

   int curl_nd, vec_nd;
   if ( trial_fe.GetMapType() == mfem::FiniteElement::H_CURL )
   {
      curl_nd = trial_nd;
      vector_dof = trial_fe.GetDof();
      vec_nd  = test_nd;
      scalar_dof = test_fe.GetDof();
      dim = trial_fe.GetDim();
      dimc = dim;
   }
   else
   {
      curl_nd = test_nd;
      vector_dof = test_fe.GetDof();
      vec_nd  = trial_nd;
      scalar_dof = trial_fe.GetDof();
      dim = test_fe.GetDim();
      dimc = dim;
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

double uFun1_ex(const Vector& x); // Exact Solution
double uFun1_ex_dt(const Vector& xt);
void uFun1_ex_gradx(const Vector& xt, Vector& grad);

void bFun_ex (const Vector& xt, Vector& b);
double  bFundiv_ex(const Vector& xt);

void bFunRect2D_ex(const Vector& xt, Vector& b );
double  bFunRect2Ddiv_ex(const Vector& xt);

void bFunCube3D_ex(const Vector& xt, Vector& b );
double  bFunCube3Ddiv_ex(const Vector& xt);

void bFunSphere3D_ex(const Vector& xt, Vector& b );
double  bFunSphere3Ddiv_ex(const Vector& xt);

void bFunCircle2D_ex (const Vector& xt, Vector& b);
double  bFunCircle2Ddiv_ex(const Vector& xt);

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

void videofun(const Vector& xt, Vector& vecvalue);

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

double cas_weight (const Vector& xt, double * params, const int &nparams);
double deletethis (const Vector& xt);

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void curlE_exact(const Vector &x, Vector &curlE);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;

// 4d test from Martin's example
void E_exactMat_vec(const Vector &x, Vector &E);
void E_exactMat(const Vector &, DenseMatrix &);
void f_exactMat(const Vector &, DenseMatrix &);


template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaTemplate(const Vector& xt, Vector& sigma);
template <void (*bvecfunc)(const Vector&, Vector& )>
    void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda);
template <void (*bvecfunc)(const Vector&, Vector& )>
    void bbTTemplate(const Vector& xt, DenseMatrix& bbT);
template <void (*bvecfunc)(const Vector&, Vector& )>
    double bTbTemplate(const Vector& xt);
template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    double minbTbSnonhomoTemplate(const Vector& xt);
template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaNonHomoTemplate(const Vector& xt, Vector& sigma);
template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& ), void (*opdivfreevec)(const Vector&, Vector& )>
        void minKsigmahatTemplate(const Vector& xt, Vector& minKsigmahatv);
template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec), void (*opdivfreevec)(const Vector&, Vector& )>
        double bsigmahatTemplate(const Vector& xt);
template<double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
        void (*opdivfreevec)(const Vector&, Vector& )>
        void sigmahatTemplate(const Vector& xt, Vector& sigmahatv);
template<double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
        void (*opdivfreevec)(const Vector&, Vector& )>
        void minsigmahatTemplate(const Vector& xt, Vector& sigmahatv);
template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
    double rhsideTemplate(const Vector& xt);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void bfTemplate(const Vector& xt, Vector& bf);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void bdivsigmaTemplate(const Vector& xt, Vector& bf);

template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec)>
        void bSnonhomoTemplate(const Vector& xt, Vector& bSnonhomo);

template<void(*bvec)(const Vector & x, Vector & vec)>
        void minbTemplate(const Vector& xt, Vector& minb);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        double divsigmaTemplate(const Vector& xt);

template<double (*S)(const Vector & xt) > double SnonhomoTemplate(const Vector& xt);

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
            //SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &cas_weight, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
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
            //SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &cas_weight, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            if (numcurl == 1)
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &zerovec_ex, &zerovec_ex>();
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
    int numsol          = 4;
    int numcurl         = 1;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 2;

    bool withDiv = true;
    bool withS = true;
    bool blockedversion = true;

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
    int sdim = nDimensions; // used in 4D case

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

#ifdef PAULINA_CODE
    if (!withDiv && verbose)
        std::cout << "Paulina's code cannot be used withut withDiv flag \n";

    int ref_levels = par_ref_levels;

    ParFiniteElementSpace *coarseR_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);
    ParFiniteElementSpace *coarseW_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

    // Input to the algorithm::

    Array< SparseMatrix*> P_W(ref_levels);
    Array< SparseMatrix*> P_R(ref_levels);
    Array< SparseMatrix*> Element_dofs_R(ref_levels);
    Array< SparseMatrix*> Element_dofs_W(ref_levels);
   // Array< int * > Sol_sig_level(ref_levels);

    const SparseMatrix* P_W_local;
    const SparseMatrix* P_R_local;

    Array<int> ess_dof_coarsestlvl_list;

    // Dofs_TrueDofs at each space:

    auto d_td_coarse_R = coarseR_space->Dof_TrueDof_Matrix();
    auto d_td_coarse_W = coarseW_space->Dof_TrueDof_Matrix();

    DivPart divp;

    for (int l = 0; l < ref_levels+1; l++){
      if (l > 0){
        //W_space->Update();
        //R_space->Update();

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

    Vector sigmahat_pau;

#else
    for (int l = 0; l < par_ref_levels; l++)
    {
        pmesh->UniformRefinement();
        if (withDiv)
             W_space->Update();
        R_space->Update();
    }
#endif
    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << "\n";

    Transport_test_divfree Mytest(nDimensions, numsol, numcurl);

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.

    shared_ptr<mfem::HypreParMatrix> A;
    HypreParMatrix Amat;
    Vector Xdebug;
    Vector X, B;
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
        hdivfree_coll = new DivSkew1_4DFECollection;
        C_space = new ParFiniteElementSpace(pmesh.get(), hdivfree_coll);

        // testing ProjectCoefficient
        VectorFunctionCoefficient * divfreepartcoeff = new
                VectorFunctionCoefficient(6, E_exactMat_vec);
        //VectorCoefficient * divfreepartcoeff = new VectorFunctionCoefficient(dim, DivmatFun4D_ex);
        ParGridFunction *u_exact = new ParGridFunction(C_space);
        u_exact->ProjectCoefficient(*divfreepartcoeff);//(*(Mytest.divfreepart));


        if (verbose)
            std::cout << "ProjectCoefficient is ok with vectors from DivSkew \n";
        //u_exact->Print();

        // checking projection error computation
        int order_quad = 2*(feorder+1) + 1;//max(2, 2*feorder+1);
        const IntegrationRule *irs[Geometry::NumGeom];
        for (int i=0; i < Geometry::NumGeom; ++i)
        {
           irs[i] = &(IntRules.Get(i, order_quad));
        }

        double norm_u = ComputeGlobalLpNorm(2, *divfreepartcoeff, *pmesh, irs);
        double projection_error_u = u_exact->ComputeL2Error(*divfreepartcoeff, irs);

        if(verbose)
        {
            std::cout << "|| u_ex - Pi_h u_ex || = " << projection_error_u << "\n";
            if ( norm_u > MYZEROTOL )
                std::cout << "|| u_ex - Pi_h u_ex || / || u_ex || = " << projection_error_u / norm_u << "\n";
            else
                std::cout << "|| Pi_h u_ex || = " << projection_error_u << " (u_ex = 0) \n ";
        }
    } // end of initialization of div-free f.e. space in 4D

    FiniteElementCollection *h1_coll;
    ParFiniteElementSpace *H_space;
    if (withS)
    {
        h1_coll = new H1_FECollection(feorder+1, nDimensions);
        H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);
    }

    int numblocks = 1;
    if (withS)
        numblocks++;

    Array<int> block_offsets(numblocks + 1); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = C_space->GetVSize();
    if (withS)
        block_offsets[2] = H_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(numblocks + 1); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = C_space->TrueVSize();
    if (withS)
        block_trueOffsets[2] = H_space->TrueVSize();
    block_trueOffsets.PartialSum();

    HYPRE_Int dimC = C_space->GlobalTrueVSize();
    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimH;
    if (withS)
        dimH = H_space->GlobalTrueVSize();
    if (verbose)
    {
       std::cout << "***********************************************************\n";
       std::cout << "dim(C) = " << dimC << "\n";
       if (withS)
       {
           std::cout << "dim(H) = " << dimH << ", ";
           std::cout << "dim(C+H) = " << dimC + dimH << "\n";
       }
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

    Array<int> ess_tdof_listU;

#ifndef PAULINA_CODE
    if (withS)
    {
        //ess_bdrU[0] = 1;
        //ess_bdrU[1] = 1;
        ess_bdrU = 1;
        ess_bdrU[pmesh->bdr_attributes.Max() - 1] = 0;
    }
    else
    {
        // correct, working
        ess_bdrU = 1;
        ess_bdrU[pmesh->bdr_attributes.Max() - 1] = 0;

        //ess_bdrU[0] = 1;
        //ess_bdrU[1] = 1;
    }
#endif
    C_space->GetEssentialTrueDofs(ess_bdrU, ess_tdof_listU);

    Array<int> ess_tdof_listS, ess_bdrS(pmesh->bdr_attributes.Max());
    ess_bdrS = 0;
    if (withS)
    {
        ess_bdrS = 0;
        ess_bdrS[0] = 1; // t = 0
        //ess_bdrS = 1;
        H_space->GetEssentialTrueDofs(ess_bdrS, ess_tdof_listS);
    }

    ParGridFunction * Sigmahat = new ParGridFunction(R_space);
    if (withDiv)
    {
        if (verbose)
            std::cout << "Assembling linear system for finding sigmahat \n";

#ifdef PAULINA_CODE
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
#else
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
#endif

    }
    else // solving a div-free system with some analytical solution for the div-free part
    {
        Sigmahat->ProjectCoefficient(*(Mytest.sigmahat));
    }

    // how to make MFEM_ASSERT working?
    //MFEM_ASSERT(dim == 3, "For now only 3D case is considered \n");
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

    ParGridFunction *S_exact;
    if (withS)
    {
        S_exact = new ParGridFunction(H_space);
        S_exact->ProjectCoefficient(*(Mytest.scalarS));
    }

    if (blockedversion || withS)
    {
        if (withDiv)
            xblks.GetBlock(0) = 0.0;
        else
            xblks.GetBlock(0) = *u_exact;
    }
    if (withS)
        xblks.GetBlock(1) = *S_exact;


    ffform = new ParLinearForm(C_space);
    if (!withDiv)
    {
        if (withS)
        {
            ffform->Update(C_space, rhsblks.GetBlock(0), 0);
            ffform->AddDomainIntegrator(new VectorcurlDomainLFIntegrator(*(Mytest.minsigmahat)));
            ffform->Assemble();
        }
        else
        {
            if (blockedversion)
                ffform->Update(C_space, rhsblks.GetBlock(0), 0);
            //else
                //ffform->Update(C_space, *u_exact, 0);
            ffform->AddDomainIntegrator(new VectorcurlDomainLFIntegrator(*(Mytest.minKsigmahat)));
            ffform->Assemble();
        }
    }
    else // if withDiv = true
    {
        if (withS)
        {
            ParMixedBilinearForm * Tblock = new ParMixedBilinearForm(R_space, C_space);
            Tblock->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator);
            Tblock->Assemble();
            Tblock->EliminateTestDofs(ess_bdrU);
            Tblock->Finalize();
            //HypreParMatrix * T = Tblock->ParallelAssemble();
            *Sigmahat *= -1.0;
            Tblock->Mult(*Sigmahat, *ffform);
            //T->Mult(*Sigmahat, *ffform);
            *Sigmahat *= -1.0;
        }
        else
        {
            //if (print_progress_report)
                //std::cout << "withHdiv is not implemented for withS = false case \n";

            if (blockedversion)
                ffform->Update(C_space, rhsblks.GetBlock(0), 0);

            ParMixedBilinearForm * Tblock = new ParMixedBilinearForm(R_space, C_space);
            Tblock->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*Mytest.Ktilda));
            Tblock->Assemble();
            Tblock->EliminateTestDofs(ess_bdrU);
            Tblock->Finalize();
            *Sigmahat *= -1.0;
            Tblock->Mult(*Sigmahat, *ffform);
            *Sigmahat *= -1.0;
        }
    }

    ParLinearForm *qform(new ParLinearForm);
    ParGridFunction * tmp_to_add;
    if (withS)
    {
        if (!withDiv)
        {
            qform->Update(H_space, rhsblks.GetBlock(1), 0);
            qform->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bdivsigma));
            qform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.bsigmahat));
            qform->Assemble();
        }
        else // Div-Div
        {
            //if (print_progress_report)
                //std::cout << "A required change of rh side qform is not implemented yet for withDiv case \n";
            qform->Update(H_space, rhsblks.GetBlock(1), 0);
            qform->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bdivsigma));
            qform->Assemble();

            ParMixedBilinearForm * LTblock = new ParMixedBilinearForm(H_space, R_space);
            LTblock->AddDomainIntegrator(new MixedVectorProductIntegrator(*(Mytest.b)));
            LTblock->Assemble();
            LTblock->EliminateTestDofs(ess_bdrS);
            //LTblock->EliminateTrialDofs(ess_bdrU);
            LTblock->Finalize();

            tmp_to_add = new ParGridFunction(H_space);
            LTblock->MultTranspose(*Sigmahat, *tmp_to_add);

            *qform += *tmp_to_add;
        }
    }

    Ablock = new ParBilinearForm(C_space);
    if (withS)
    {
        Coefficient *one = new ConstantCoefficient(1.0);
        Ablock->AddDomainIntegrator(new CurlCurlIntegrator(*one));
        Ablock->Assemble();
        Ablock->EliminateEssentialBC(ess_bdrU,xblks.GetBlock(0),*ffform);
        Ablock->Finalize();
        HypreParMatrix * tempA = Ablock->ParallelAssemble();
        A = make_shared<HypreParMatrix>(*tempA);
    }
    else //if (!withS)
    {
        Ablock->AddDomainIntegrator(new CurlCurlIntegrator(*(Mytest.Ktilda)));
        Ablock->Assemble();
        if (blockedversion)
        {
            Ablock->EliminateEssentialBC(ess_bdrU,xblks.GetBlock(0),*ffform);
            Ablock->Finalize();
            HypreParMatrix * tempA = Ablock->ParallelAssemble();
            A = make_shared<HypreParMatrix>(*tempA);
        }
    }

    ParBilinearForm *Cblock;
    HypreParMatrix *C;

    if (withS)
    {
        Cblock = new ParBilinearForm(H_space);
        Cblock->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));
        Cblock->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.bbT));
        Cblock->Assemble();
        Cblock->EliminateEssentialBC(ess_bdrS, xblks.GetBlock(1),*qform);
        Cblock->Finalize();
        C = Cblock->ParallelAssemble();
    }

    ParMixedBilinearForm *CHblock;
    HypreParMatrix *CH, *CHT;

    if (withS)
    {
        CHblock = new ParMixedBilinearForm(C_space, H_space);
        CHblock->AddDomainIntegrator(new VectorFECurlVQIntegrator(*Mytest.minb));
        CHblock->Assemble();
        CHblock->EliminateTestDofs(ess_bdrS);

        CHblock->EliminateTrialDofs(ess_bdrU, xblks.GetBlock(0), *qform);

        CHblock->Finalize();
        CH = CHblock->ParallelAssemble();
        CHT = CH->Transpose();
    }

    if (blockedversion || withS)
    {
        ffform->ParallelAssemble(trueRhs.GetBlock(0));
    }
    if (withS)
    {
        qform->ParallelAssemble(trueRhs.GetBlock(1));
    }

    BlockOperator *MainOp = new BlockOperator(block_trueOffsets);

    if (withS)
    {
        MainOp->SetBlock(0,0, A.get());
        MainOp->SetBlock(0,1, CHT);
        MainOp->SetBlock(1,0, CH);
        MainOp->SetBlock(1,1, C);
    }
    else
    {
        if (blockedversion)
            MainOp->SetBlock(0,0, A.get());
        else
        {
            Ablock->FormLinearSystem(ess_tdof_listU, *u_exact, *ffform, Amat, Xdebug, B);
            MainOp->SetBlock(0,0, &Amat);
        }
    }

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
                if (withS) // case of block system
                {
                    prec = new BlockDiagonalPreconditioner(block_trueOffsets);
                    Operator * precU = new HypreAMS(*A, C_space);
                    ((HypreAMS*)precU)->SetSingularProblem();
                    Operator * precS = new HypreBoomerAMG(*C);
                    ((HypreBoomerAMG*)precS)->SetPrintLevel(0);

                    ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, precU);
                    ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(1, precS);
                }
                else // only equation in div-free subspace
                {
                    if (blockedversion)
                        prec = new HypreAMS(*A, C_space);
                    else
                        prec = new HypreAMS(Amat, C_space);
                    ((HypreAMS*)prec)->SetSingularProblem();
                }

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

    if (!withS && !blockedversion)
    {
        X.SetSize(MainOp->Height());
        X = 0.0;
    }

    if (with_prec)
        solver->SetPreconditioner(*prec);
    solver->SetPrintLevel(0);
    if (withS || blockedversion )
    {
        trueX = 0.0;
        solver->Mult(trueRhs, trueX);
    }
    else
        solver->Mult(B, X);
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
    if (blockedversion || withS)
    {
        u->Distribute(&(trueX.GetBlock(0)));
        if (withS)
        {
            S = new ParGridFunction(H_space);
            S->Distribute(&(trueX.GetBlock(1)));
        }
    }
    else
        Ablock->RecoverFEMSolution(X, *ffform, *u);

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
        if ( norm_u > MYZEROTOL )
        {
            //std::cout << "norm_u = " << norm_u << "\n";
            cout << "|| u - u_ex || / || u_ex || = " << err_u / norm_u << endl;
        }
        else
            cout << "|| u || = " << err_u << " (u_ex = 0)" << endl;

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
        if ( norm_divfreepart > MYZEROTOL )
        {
            //cout << "|| divfreepart_ex || = " << norm_divfreepart << endl;
            cout << "|| curl_h u_h - divfreepart_ex || / || divfreepart_ex || = " << err_divfreepart / norm_divfreepart << endl;
        }
        else
            cout << "|| curl_h u_h || = " << err_divfreepart << " (divfreepart_ex = 0)" << endl;

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
        if ( norm_sigma > MYZEROTOL )
            cout << "|| sigma_h - sigma_ex || / || sigma_ex || = " << err_sigma / norm_sigma << endl;
        else
            cout << "|| sigma || = " << err_sigma << " (sigma_ex = 0)" << endl;

    double err_sigmahat = Sigmahat->ComputeL2Error(*(Mytest.sigma), irs);

    /*
    if (verbose && !withDiv)
        if ( norm_sigma > MYZEROTOL )
            cout << "|| sigma_hat - sigma_ex || / || sigma_ex || = " << err_sigmahat / norm_sigma << endl;
        else
            cout << "|| sigma_hat || = " << err_sigmahat << " (sigma_ex = 0)" << endl;
    */

    double norm_S;
    if (withS)
    {
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
    }


    if (verbose)
        cout << "Computing projection errors" << endl;

    //double projection_error_u = u_exact->ComputeL2Error(E, irs);
    double projection_error_u = u_exact->ComputeL2Error(*(Mytest.divfreepart), irs);

    if(verbose && !withDiv)
        if ( norm_u > MYZEROTOL )
        {
            //std::cout << "Debug: || u_ex || = " << norm_u << "\n";
            //std::cout << "Debug: proj error = " << projection_error_u << "\n";
            cout << "|| u_ex - Pi_h u_ex || / || u_ex || = " << projection_error_u / norm_u << endl;
        }
        else
            cout << "|| Pi_h u_ex || = " << projection_error_u << " (u_ex = 0) \n ";

    double projection_error_sigma = sigma_exact->ComputeL2Error(*(Mytest.sigma), irs);

    if(verbose)
        if ( norm_sigma > MYZEROTOL )
        {
            cout << "|| sigma_ex - Pi_h sigma_ex || / || sigma_ex || = " << projection_error_sigma / norm_sigma << endl;
        }
        else
            cout << "|| Pi_h sigma_ex || = " << projection_error_sigma << " (sigma_ex = 0) \n ";
    if (withS)
    {
        double projection_error_S = S_exact->ComputeL2Error(*(Mytest.scalarS), irs);

         if(verbose)
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

       if (withS)
       {
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
        }

       MPI_Barrier(pmesh->GetComm());
    }

    // 17. Free the used memory.
    delete ffform;
    if (withS)
        delete qform;

    if (withS || blockedversion)
    {
        delete Ablock;
        if (withS)
        {
            delete Cblock;
            delete CHblock;
        }
    }

    delete C_space;
    delete hdivfree_coll;
    delete R_space;
    delete hdiv_coll;
    if (withS)
    {
       delete H_space;
       delete h1_coll;
    }

    MPI_Finalize();
    return 0;
}

template <void (*bvecfunc)(const Vector&, Vector& )> \
void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda)
{
    int nDimensions = xt.Size();
    Ktilda.SetSize(nDimensions);
    Vector b;
    bvecfunc(xt,b);
    double bTbInv = (-1./(b*b));
    Ktilda.Diag(1.0,nDimensions);
#ifndef K_IDENTITY
    AddMult_a_VVt(bTbInv,b,Ktilda);
#endif
}

template <void (*bvecfunc)(const Vector&, Vector& )> \
void bbTTemplate(const Vector& xt, DenseMatrix& bbT)
{
    Vector b;
    bvecfunc(xt,b);
    MultVVt(b, bbT);
}


template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )> \
void sigmaTemplate(const Vector& xt, Vector& sigma)
{
    Vector b;
    bvecfunc(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = S(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);

    return;
}

template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )> \
void sigmaNonHomoTemplate(const Vector& xt, Vector& sigma) // sigmaNonHomo = (b S0, S0)^T for S0 = S(t=0)
{
    sigma.SetSize(xt.Size());

    Vector xteq0;
    xteq0.SetSize(xt.Size()); // xt with t = 0
    xteq0 = xt;
    xteq0(xteq0.Size()-1) = 0.0;

    sigmaTemplate<S, bvecfunc>(xteq0, sigma);
/*
    Vector b;
    bvecfunc(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = ufunc(xteq0);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);
*/
    return;
}


template <void (*bvecfunc)(const Vector&, Vector& )> \
double bTbTemplate(const Vector& xt)
{
    Vector b;
    bvecfunc(xt,b);
    return b*b;
}

template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )> \
double minbTbSnonhomoTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return - bTbTemplate<bvecfunc>(xt) * S(xt0);
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
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
double rhsideTemplate(const Vector& xt)
{
    Vector b;
    bvec(xt,b);

    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    Vector gradS0;
    Sgradxvec(xt0,gradS0);

    double res = divsigmaTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>(xt);

    for ( int i= 0; i < xt.Size() - 1; ++i )
        res += b(i) * (- gradS0(i));
    res += divbfunc(xt) * ( - S(xt0));

    return res;
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
void bfTemplate(const Vector& xt, Vector& bf)
{
    bf.SetSize(xt.Size());

    Vector b;
    bvec(xt,b);

    double f = rhsideTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>(xt);

    for (int i = 0; i < bf.Size(); ++i)
        bf(i) = f * b(i);
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
void bdivsigmaTemplate(const Vector& xt, Vector& bdivsigma)
{
    bdivsigma.SetSize(xt.Size());

    Vector b;
    bvec(xt,b);

    double divsigma = divsigmaTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>(xt);

    for (int i = 0; i < bdivsigma.Size(); ++i)
        bdivsigma(i) = divsigma * b(i);
}


template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec)>
void bSnonhomoTemplate(const Vector& xt, Vector& bSnonhomo)
{
    bSnonhomo.SetSize(xt.Size());

    Vector b;
    bvec(xt,b);

    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    for (int i = 0; i < bSnonhomo.Size(); ++i)
        bSnonhomo(i) = S(xt0) * b(i);
}

template<void(*bvec)(const Vector & x, Vector & vec)>
void minbTemplate(const Vector& xt, Vector& minb)
{
    minb.SetSize(xt.Size());

    bvec(xt,minb);

    minb *= -1;
}

template<double (*S)(const Vector & xt) > double SnonhomoTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return S(xt0);
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
    //return t * exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y)) + 5.0 * (x + y);
}

double uFun4_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return uFun4_ex(xt);
    //return (1 + t) * exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y));
}

void uFun4_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y));
    gradx(1) = exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y));
    //gradx(0) = t * exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y)) + 5.0;
    //gradx(1) = t * exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y)) + 5.0;
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
    vecvalue(0) = sin(kappa * xt(1));
    vecvalue(1) = sin(kappa * xt(2));
    vecvalue(2) = sin(kappa * xt(0));

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
    vecvalue(0) = - kappa * cos(kappa * xt(2));
    vecvalue(1) = - kappa * cos(kappa * xt(0));
    vecvalue(2) = - kappa * cos(kappa * xt(1));

    return;
}

////////////////
void DivmatFun4D_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    // 4D counterpart of the Martin's 3D function
    //std::cout << "Error: DivmatFun4D_ex is incorrect \n";
    vecvalue(0) = sin(kappa * xt(1));
    vecvalue(1) = sin(kappa * xt(2));
    vecvalue(2) = sin(kappa * xt(3));
    vecvalue(3) = sin(kappa * xt(0));

    return;
}

void DivmatDivmatFun4D_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //vecvalue(0) = 0.0;
    //vecvalue(1) = 0.0;
    //vecvalue(2) = -2.0 * (1 - t);

    // Divmat of the 4D counterpart of the Martin's 3D function
    std::cout << "Error: DivmatDivmatFun4D_ex is incorrect \n";
    vecvalue(0) = - kappa * cos(kappa * xt(2));
    vecvalue(1) = - kappa * cos(kappa * xt(0));
    vecvalue(2) = - kappa * cos(kappa * xt(1));

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

template <double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void minKsigmahatTemplate(const Vector& xt, Vector& minKsigmahatv)
{
    minKsigmahatv.SetSize(xt.Size());

    Vector b;
    bvecfunc(xt, b);

    Vector sigmahatv;
    sigmahatTemplate<S, bvecfunc, opdivfreevec>(xt, sigmahatv);

    DenseMatrix Ktilda;
    KtildaTemplate<bvecfunc>(xt, Ktilda);

    Ktilda.Mult(sigmahatv, minKsigmahatv);

    minKsigmahatv *= -1.0;
    return;
}

template <double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
double bsigmahatTemplate(const Vector& xt)
{
    Vector b;
    bvecfunc(xt, b);

    Vector sigmahatv;
    sigmahatTemplate<S, bvecfunc, opdivfreevec>(xt, sigmahatv);

    return b * sigmahatv;
}

template <double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void sigmahatTemplate(const Vector& xt, Vector& sigmahatv)
{
    sigmahatv.SetSize(xt.Size());

    Vector b;
    bvecfunc(xt, b);

    Vector sigma(xt.Size());
    sigma(xt.Size()-1) = S(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);

    Vector opdivfree;
    opdivfreevec(xt, opdivfree);

    sigmahatv = 0.0;
    sigmahatv -= opdivfree;
#ifndef ONLY_DIVFREEPART
    sigmahatv += sigma;
#endif
    return;
}

template <double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void minsigmahatTemplate(const Vector& xt, Vector& minsigmahatv)
{
    minsigmahatv.SetSize(xt.Size());
    sigmahatTemplate<S, bvecfunc, opdivfreevec>(xt, minsigmahatv);
    minsigmahatv *= -1;

    return;
}

void E_exact(const Vector &xt, Vector &E)
{
   if (xt.Size() == 3)
   {

       E(0) = sin(kappa * xt(1));
       E(1) = sin(kappa * xt(2));
       E(2) = sin(kappa * xt(0));
#ifdef BAD_TEST
       double x = xt(0);
       double y = xt(1);
       double t = xt(2);

       E(0) = x * x * (1-x) * (1-x) * y * y * (1-y) * (1-y) * t * t * (1-t) * (1-t);
       E(1) = 0.0;
       E(2) = 0.0;
#endif
   }
}


void curlE_exact(const Vector &xt, Vector &curlE)
{
   if (xt.Size() == 3)
   {
       curlE(0) = - kappa * cos(kappa * xt(2));
       curlE(1) = - kappa * cos(kappa * xt(0));
       curlE(2) = - kappa * cos(kappa * xt(1));
#ifdef BAD_TEST
       double x = xt(0);
       double y = xt(1);
       double t = xt(2);

       curlE(0) = 0.0;
       curlE(1) =  2.0 * t * (1-t) * (1.-2.*t) * x * x * (1-x) * (1-x) * y * y * (1-y) * (1-y);
       curlE(2) = -2.0 * y * (1-y) * (1.-2.*y) * x * x * (1-x) * (1-x) * t * t * (1-t) * (1-t);
#endif
   }
}


void vminusone_exact(const Vector &x, Vector &vminusone)
{
   vminusone.SetSize(x.Size());
   vminusone = -1.0;
}

void vone_exact(const Vector &x, Vector &vone)
{
   vone.SetSize(x.Size());
   vone = 1.0;
}


void f_exact(const Vector &xt, Vector &f)
{
   if (xt.Size() == 3)
   {


       //f(0) = sin(kappa * x(1));
       //f(1) = sin(kappa * x(2));
       //f(2) = sin(kappa * x(0));
       //f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
       //f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
       //f(2) = (1. + kappa * kappa) * sin(kappa * x(0));

       f(0) = kappa * kappa * sin(kappa * xt(1));
       f(1) = kappa * kappa * sin(kappa * xt(2));
       f(2) = kappa * kappa * sin(kappa * xt(0));

       /*

       double x = xt(0);
       double y = xt(1);
       double t = xt(2);

       f(0) =  -1.0 * (2 * (1-y)*(1-y) + 2*y*y - 2.0 * 2 * y * 2 * (1-y)) * x * x * (1-x) * (1-x) * t * t * (1-t) * (1-t);
       f(0) += -1.0 * (2 * (1-t)*(1-t) + 2*t*t - 2.0 * 2 * t * 2 * (1-t)) * x * x * (1-x) * (1-x) * y * y * (1-y) * (1-y);
       f(1) = 2.0 * y * (1-y) * (1-2*y) * 2.0 * x * (1-x) * (1-2*x) * t * t * (1-t) * (1-t);
       f(2) = 2.0 * t * (1-t) * (1-2*t) * 2.0 * x * (1-x) * (1-2*x) * y * y * (1-y) * (1-y);
       */


   }
}


void E_exactMat_vec(const Vector &x, Vector &E)
{
   int dim = x.Size();

   if (dim==4)
   {
      E.SetSize(6);

      double s0 = sin(M_PI*x(0)), s1 = sin(M_PI*x(1)), s2 = sin(M_PI*x(2)),
             s3 = sin(M_PI*x(3));
      double c0 = cos(M_PI*x(0)), c1 = cos(M_PI*x(1)), c2 = cos(M_PI*x(2)),
             c3 = cos(M_PI*x(3));

      E(0) =  c0*c1*s2*s3;
      E(1) = -c0*s1*c2*s3;
      E(2) =  c0*s1*s2*c3;
      E(3) =  s0*c1*c2*s3;
      E(4) = -s0*c1*s2*c3;
      E(5) =  s0*s1*c2*c3;
   }
}

void E_exactMat(const Vector &x, DenseMatrix &E)
{
   int dim = x.Size();

   E.SetSize(dim*dim);

   if (dim==4)
   {
      Vector vecE;
      E_exactMat_vec(x, vecE);

      E = 0.0;

      E(0,1) = vecE(0);
      E(0,2) = vecE(1);
      E(0,3) = vecE(2);
      E(1,2) = vecE(3);
      E(1,3) = vecE(4);
      E(2,3) = vecE(5);

      E(1,0) =  -E(0,1);
      E(2,0) =  -E(0,2);
      E(3,0) =  -E(0,3);
      E(2,1) =  -E(1,2);
      E(3,1) =  -E(1,3);
      E(3,2) =  -E(2,3);
   }
}



//f_exact = E + 0.5 * P( curl DivSkew E ), where P is the 4d permutation operator
void f_exactMat(const Vector &x, DenseMatrix &f)
{
   int dim = x.Size();

   f.SetSize(dim,dim);

   if (dim==4)
   {
      f = 0.0;

      double s0 = sin(M_PI*x(0)), s1 = sin(M_PI*x(1)), s2 = sin(M_PI*x(2)),
             s3 = sin(M_PI*x(3));
      double c0 = cos(M_PI*x(0)), c1 = cos(M_PI*x(1)), c2 = cos(M_PI*x(2)),
             c3 = cos(M_PI*x(3));

      f(0,1) =  (1.0 + 1.0  * M_PI*M_PI)*c0*c1*s2*s3;
      f(0,2) = -(1.0 + 0.0  * M_PI*M_PI)*c0*s1*c2*s3;
      f(0,3) =  (1.0 + 1.0  * M_PI*M_PI)*c0*s1*s2*c3;
      f(1,2) =  (1.0 - 1.0  * M_PI*M_PI)*s0*c1*c2*s3;
      f(1,3) = -(1.0 + 0.0  * M_PI*M_PI)*s0*c1*s2*c3;
      f(2,3) =  (1.0 + 1.0  * M_PI*M_PI)*s0*s1*c2*c3;

      f(1,0) =  -f(0,1);
      f(2,0) =  -f(0,2);
      f(3,0) =  -f(0,3);
      f(2,1) =  -f(1,2);
      f(3,1) =  -f(1,3);
      f(3,2) =  -f(2,3);
   }
}

double uFun1_ex(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    //return t;
    ////setback
    return sin(t)*exp(t);
}

double uFun1_ex_dt(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    return (cos(t) + sin(t)) * exp(t);
}

void uFun1_ex_gradx(const Vector& xt, Vector& gradx )
{
    gradx.SetSize(xt.Size() - 1);
    gradx = 0.0;
}
