#include "mfem.hpp"

using namespace mfem;
using namespace std;
using std::unique_ptr;

#define DEBUG_INFO
#define OLDFASHION


// Computes and prints the norm of || Funct * y ||_2,h
void CheckFunctValue(const BlockMatrix& Funct, const BlockVector& yblock, char * string)
{
    BlockVector res(Funct.ColOffsets());
    Funct.Mult(yblock, res);
    double func_norm = res.Norml2() / sqrt (res.Size());
    std::cout << "Functional norm " << string << func_norm << " ... \n";
}

// Computes and prints the norm of || Constr * sigma - ConstrRhs ||_2,h
void CheckConstrRes(Vector& sigma, const SparseMatrix& Constr, const Vector& ConstrRhs,
                                                const char* string)
{
    Vector res_constr(Constr.Height());
    Constr.Mult(sigma, res_constr);
    //ofstream ofs("newsolver_out.txt");
    //res_constr.Print(ofs,1);
    res_constr -= ConstrRhs;
    double constr_norm = res_constr.Norml2() / sqrt (res_constr.Size());
    std::cout << "Constraint residual norm " << string << ": " << constr_norm << " ... \n";
}

// TODO: Add blas and lapack versions for solving local problems
// TODO: Test after all againt the case with nonzero boundary conditions for sigma
// FIXME: Projecting solutions from the finer levels each time immediately
// to the finest level is redundant

class BaseGeneralMinConstrSolver : public Solver
{
protected:
    int num_levels;
    // iteration index (solver behavior is different for the first iteration)
    int * current_iterate;
    const Array< SparseMatrix*>& AE_e;
    const Array< BlockMatrix*>& el_to_dofs_Func;
    const Array< SparseMatrix*>& el_to_dofs_L2;
    const std::vector<HypreParMatrix*>& dof_trueDof_Func;
    const HypreParMatrix& dof_trueDof_L2;
    const Array< BlockMatrix*>& P_Func;
    const Array< SparseMatrix*>& P_L2;
    // for each variable in the functional and for each level stores a boolean
    // vector which defines if a dof is at the boundary
    const std::vector<Array<Array<int>* > > & bdrdofs_Func;
    const BlockMatrix& Funct;
    const Array<int>& block_offsets;
    const int numblocks;
    const SparseMatrix& Constr;
    const Vector& ConstrRhs;
    const BlockVector& bdrdata_finest;

    const Array<int>& ess_dof_coarsest;

    bool higher_order;


    // temporary variables
    // FIXME: is it a good practice? should they be mutable?
    // Have to use pointers everywhere because Solver::Mult() must not change the solver data members
    mutable Array<SparseMatrix*> *AE_edofs_L2;
    mutable Array<BlockMatrix*> *AE_eintdofs_Func; // relation between AEs and internal (w.r.t to AEs) fine-grid dofs

    // stores Functional matrix on all levels except the finest
    // so that (*Funct_levels)[0] = Functional matrix on level 1 (not level 0!)
    mutable Array<BlockMatrix*> *Funct_lvls;
    mutable Array<SparseMatrix*> *Constr_lvls;

    // storage for prerequisites of the coarsest level problem: offsets, matrix and preconditoner
    mutable Array<int>* coarse_offsets;
    mutable BlockOperator* coarse_matrix;
    mutable BlockDiagonalPreconditioner * coarse_prec;

    mutable BlockVector* xblock; // temporary variables for casting (sigma,s) vectors into proper block vectors
    mutable BlockVector* yblock;

    // variable-size vectors (initialized with the finest level sizes)
    mutable Vector* rhs_constr;     // righthand side (from the divergence constraint) at level l
    mutable Vector* Qlminus1_f;
    mutable Vector* workfvec;

    // used for storing solution updates at all levels
    mutable Array<BlockVector*> *solupdate_lvls;

    // temporary storage for blockvectors related to the considered functional at all levels
    // initialized in the constructor (partly) and in SetUpFinerLvl()
    // used at least in InterpolateBack() and SolveLocalProblems()
    mutable Array<BlockVector*> * tempvec_lvls;
    mutable Array<BlockVector*> * rhsfunc_lvls;

protected:
    BlockMatrix* Get_AE_eintdofs(int level, BlockMatrix& el_to_dofs, const std::vector<Array<Array<int> *> > &dof_is_bdr) const;
    void ProjectFinerL2ToCoarser(int level, const Vector& in, Vector &ProjTin, Vector &out) const;
    void ProjectFinerFuncToCoarser(int level, const BlockVector& in, BlockVector& out) const;
    void InterpolateBack(int start_level, BlockVector &vec_start, int end_level, BlockVector &vec_end) const;

    // It is virtual because one might want a complicated strategy where
    // e.g., if there are sigma and S in the functional, but each iteration
    // minimization is done only over one of the variables, thus requiring
    // rhs computation more complicated than just a simple matvec
    virtual void ComputeRhsFunc(BlockVector& rhs_func, const BlockVector& xblock) const;
    void ComputeLocalRhsConstr(int level) const;
    virtual void SetUpFinerLvl(int lvl) const;
    virtual void SetUpCoarsestLvl() const;
    void ComputeNextLvlRhsFunc(int level) const;
    void SolveLocalProblems(int level, BlockVector &lvlrhs_func, Vector& rhs_constr, BlockVector& sol_update) const;
    // These are the main differences between possible inheriting classes
    // since they define the way how the local problems are solved
    virtual void SolveLocalProblem(std::vector<DenseMatrix> &FunctBlks, DenseMatrix& B,
                                   BlockVector &G, Vector& F, BlockVector &sol, bool is_degenerate) const = 0;
    virtual void SolveCoarseProblem(BlockVector &coarserhs_func, Vector& rhs_constr, BlockVector& sol_coarse) const = 0;
public:
    // constructor
    BaseGeneralMinConstrSolver(int NumLevels,
                           const Array< SparseMatrix*> &AE_to_e,
                           const Array< BlockMatrix*> &El_to_dofs_Func, const Array< SparseMatrix*> &El_to_dofs_L2,
                           const std::vector<HypreParMatrix*>& Dof_TrueDof_Func, const HypreParMatrix& Dof_TrueDof_L2,
                           const Array< BlockMatrix*> &Proj_Func, const Array< SparseMatrix*> &Proj_L2,
                           const std::vector<Array<Array<int> *> > &BdrDofs_Func,
                           const BlockMatrix& FunctBlockMat,
                           const SparseMatrix& ConstrMat, const Vector& ConstrRhsVec,
                           const BlockVector& Bdrdata_Finest,
                           const Array<int>& Ess_dof_coarsest,
                           bool Higher_Order_Elements = false);
    BaseGeneralMinConstrSolver() = delete;

    // existence of these methods is required by the (abstract) base class Solver
    virtual void Mult(const Vector & x, Vector & y) const;
    virtual void SetOperator(const Operator &op){}
    // main top-level solving routine
    void Solve(BlockVector &previous_sol, BlockVector &next_sol) const;
};

BaseGeneralMinConstrSolver::BaseGeneralMinConstrSolver(int NumLevels,
                       const Array< SparseMatrix*> &AE_to_e,
                       const Array< BlockMatrix*> &El_to_dofs_Func, const Array< SparseMatrix*> &El_to_dofs_L2,
                       const std::vector<HypreParMatrix*>& Dof_TrueDof_Func, const HypreParMatrix& Dof_TrueDof_L2,
                       const Array< BlockMatrix*> &Proj_Func, const Array< SparseMatrix*> &Proj_L2,
                       const std::vector<Array<Array<int>* > > &BdrDofs_Func,
                       const BlockMatrix& FunctBlockMat,
                       const SparseMatrix& ConstrMat, const Vector& ConstrRhsVec,
                       const BlockVector& Bdrdata_Finest,
                       const Array<int>& Ess_dof_coarsest,
                       bool Higher_Order_Elements)
     : Solver(), num_levels(NumLevels),
       AE_e(AE_to_e),
       el_to_dofs_Func(El_to_dofs_Func), el_to_dofs_L2(El_to_dofs_L2),
       dof_trueDof_Func(Dof_TrueDof_Func), dof_trueDof_L2(Dof_TrueDof_L2),
       P_Func(Proj_Func), P_L2(Proj_L2),
       bdrdofs_Func(BdrDofs_Func),
       Funct(FunctBlockMat),
       block_offsets(Funct.RowOffsets()),
       numblocks(Funct.NumColBlocks()),
       Constr(ConstrMat),
       ConstrRhs(ConstrRhsVec),
       bdrdata_finest(Bdrdata_Finest),
       ess_dof_coarsest(Ess_dof_coarsest),
       higher_order(Higher_Order_Elements)
{
    AE_edofs_L2 = new Array<SparseMatrix*>(num_levels - 1);
    AE_eintdofs_Func = new Array<BlockMatrix*>(num_levels - 1);
    rhs_constr = new Vector(Constr.Height());
    Qlminus1_f = new Vector(Constr.Height());
    workfvec = new Vector(Constr.Height());
    xblock = new BlockVector(block_offsets);
    yblock = new BlockVector(block_offsets);
    current_iterate = new int[1];
    *current_iterate = 0;
    Funct_lvls = new Array<BlockMatrix*>(num_levels - 1);
    Constr_lvls = new Array<SparseMatrix*>(num_levels - 1);
    tempvec_lvls = new Array<BlockVector*>(num_levels);
    tempvec_lvls[0] = new BlockVector(Funct.RowOffsets());
    rhsfunc_lvls = new Array<BlockVector*>(num_levels);
    rhsfunc_lvls[0] = new BlockVector(block_offsets);
    solupdate_lvls = new Array<BlockVector*>(num_levels);
    solupdate_lvls[0] = new BlockVector(block_offsets);
}

void BaseGeneralMinConstrSolver::Mult(const Vector & x, Vector & y) const
{
    xblock->Update(x.GetData(), block_offsets);
    yblock->Update(y.GetData(), block_offsets);
    Solve(*xblock, *yblock);

#ifdef DEBUG_INFO
    CheckFunctValue(Funct, *yblock, "at the end of iteration: ");
    CheckConstrRes(yblock->GetBlock(0), Constr, ConstrRhs, "at the end of iteration");
    /*
    Vector res_constr(Constr.Height());
    Constr.Mult(yblock->GetBlock(0), res_constr);
    res_constr -= ConstrRhs;
    double constr_norm = res_constr.Norml2() / sqrt (res_constr.Size());
    std::cout << "Constraint residual norm at the end of iteration: " << constr_norm << " ... \n";
    */
#endif
}

// Computes rhs coming from the last iterate sigma
// rhs_func = - A * x, where A is the matrix arising
// from the local minimization functional, and x is the
// minimzed variable (sigma).
// FIXME: one-liner?
void BaseGeneralMinConstrSolver::ComputeRhsFunc(BlockVector &rhs_func, const BlockVector& xblock) const
{
    Funct.Mult(xblock, rhs_func);
    *rhs_func *= -1.0;
}

// FIXME: one-liner?
void BaseGeneralMinConstrSolver::ProjectFinerFuncToCoarser(int level, const BlockVector& in, BlockVector& out) const
{
    P_Func[level]->MultTranspose(in, out);
}


// Computes rhs in the functional part for the next level
// rhs_{l+1} = P_l^T ( rhs_l - M_l sol_l )
// (*) Uses tempvec_lvls as intermediate buffers
void BaseGeneralMinConstrSolver::ComputeNextLvlRhsFunc(int level) const
{
    ProjectFinerFuncToCoarser(level, *((*rhsfunc_lvls)[level]), *((*rhsfunc_lvls)[level + 1]));

    (*Funct_lvls)[level]->Mult(*((*solupdate_lvls)[level]),*((*tempvec_lvls)[level]) );
    ProjectFinerFuncToCoarser(level, *((*tempvec_lvls)[level]), *((*tempvec_lvls)[level + 1]));

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        (*rhsfunc_lvls)[level + 1]->GetBlock(blk) -= (*tempvec_lvls)[level + 1]->GetBlock(blk);
    }

}

void BaseGeneralMinConstrSolver::Solve(BlockVector& previous_sol, BlockVector& next_sol) const
{
#ifdef DEBUG_INFO
    std::cout << "Starting iteration " << *current_iterate << " ... \n";
#endif
    // 0. preliminaries

    if (*current_iterate == 0) // initializing solution with the given boundary data
    {
        // ensure that the initial iterate satisfies essential boundary conditions
        for ( int blk = 0; blk < numblocks; ++blk)
            previous_sol.GetBlock(blk) = bdrdata_finest.GetBlock(blk);
#ifdef DEBUG_INFO
        CheckFunctValue(Funct, previous_sol, "for prev_sol at the beginning of iteration 0: ");
#endif
    }

    ComputeRhsFunc(*((*rhsfunc_lvls)[0]), previous_sol);

    for ( int blk = 0; blk < numblocks; ++blk)
        next_sol.GetBlock(blk) = previous_sol.GetBlock(blk);

#ifdef DEBUG_INFO
    if (*current_iterate > 0)
    {
        CheckFunctValue(Funct, next_sol, "for next_sol at the beginning of iteration 0: ");
    }
#endif

    if (*current_iterate == 0) // for the first iteration rhs in the constraint is nonzero
    {
        // setting rhs in the constraint to the input rhs - Constr * given bdr_data
        Constr.Mult(previous_sol.GetBlock(0), *rhs_constr);
        *rhs_constr *= -1.0;
        *rhs_constr += ConstrRhs;

#ifdef DEBUG_INFO
        if (*current_iterate > 0 && rhs_constr->Norml2() / sqrt(rhs_constr->Size()) > 1.0e-13)
            std::cout << "Error! Rhs in the contraint must be 0 for iteration "
                      << *current_iterate << " > 0!\n";
#endif
    }

    *Qlminus1_f = *rhs_constr;

    // 1. loop over levels finer than the coarsest
    for (int l = 0; l < num_levels - 1; ++l)
    {
        // solution updates will always satisfy homogeneous essential boundary conditions
        for ( int blk = 0; blk < numblocks; ++blk)
            (*solupdate_lvls)[l]->GetBlock(blk) = 0.0;

        // at the first iteration we need to compute righthand side
        // for the current level and set up next level data
        // (i.e allocate memory, compute coarser matrices)
        if (*current_iterate == 0)
        {
            SetUpFinerLvl(l);
            ComputeLocalRhsConstr(l);
        }

        // solve local problems at level l
        // FIXME: all factors of local matrices can be stored after the first solver iteration
        SolveLocalProblems(l, *((*rhsfunc_lvls)[l]), *rhs_constr, *((*solupdate_lvls)[l]));

        // setting up rhs from the functional for the next (coarser) level
        ComputeNextLvlRhsFunc(l);
        //if (l == 0)
        //*((*rhsfunc_lvls)[l+1]) = 0.0;

    } // end of loop over finer levels

    // 2. setup and solve the coarse problem
    if (*current_iterate == 0)
        *rhs_constr = *Qlminus1_f;
    else
        rhs_constr->SetSize((*Constr_lvls)[num_levels - 1 - 1]->Height());

    // needs to have coarse level rhs in the func already set before the call
    SetUpCoarsestLvl();

#ifdef DEBUG_INFO
    ofstream ofs("newsolver_constr_coarserhs.txt");
    rhs_constr->Print(ofs,1);
#endif

    // 2.5 solve coarse problem
    SolveCoarseProblem(*((*rhsfunc_lvls)[num_levels-1]), *rhs_constr, *((*solupdate_lvls)[num_levels-1]));

    // 3. assemble the final solution update
#ifdef OLDFASHION
    for (int level = 0; level < num_levels; ++level)
    {
        std::cout << "level " << level << " update: \n";
        if (level == 0)
        {
            next_sol += *((*solupdate_lvls)[0]);
            if (*current_iterate == 0)
                CheckConstrRes((*solupdate_lvls)[0]->GetBlock(0), Constr, ConstrRhs, "only for the level solution");
            else
            {
                Vector zeros(ConstrRhs.Size());
                zeros = 0.0;
                CheckConstrRes((*solupdate_lvls)[0]->GetBlock(0), Constr, zeros, "only for the level solution");
            }
            ofstream ofs("newsolver_out_sol_level_0.txt");
            next_sol.GetBlock(0).Print(ofs,1);
        }
        else
        {
            InterpolateBack(level, *((*solupdate_lvls)[level]), 0, *((*tempvec_lvls)[0]));
            if (*current_iterate == 0)
                CheckConstrRes((*tempvec_lvls)[0]->GetBlock(0), Constr, ConstrRhs, "only for the level solution");
            else
            {
                Vector zeros(ConstrRhs.Size());
                zeros = 0.0;
                CheckConstrRes((*tempvec_lvls)[0]->GetBlock(0), Constr, zeros, "only for the level solution");
            }
            next_sol += *((*tempvec_lvls)[0]);
            if (level == 1)
            {
                ofstream ofs("newsolver_out_sol_level_1.txt");
                (*tempvec_lvls)[0]->GetBlock(0).Print(ofs,1);
                Vector res_constr((*Constr_lvls)[level - 1]->Height());
                (*Constr_lvls)[level - 1]->Mult((*solupdate_lvls)[level]->GetBlock(0), res_constr);
                ofstream ofs2("newsolver_out_right_at_level1.txt");
                res_constr.Print(ofs2,1);
            }
            if (level == num_levels - 1)
            {
                ofstream ofs("newsolver_out_sol_level_coarse.txt");
                (*tempvec_lvls)[0]->GetBlock(0).Print(ofs,1);
                ofstream ofs2("newsolver_out_sol_level_coarse_at_coarse.txt");
                (*solupdate_lvls)[level]->GetBlock(0).Print(ofs2,1);
            }
        }

        CheckConstrRes(next_sol.GetBlock(0), Constr, ConstrRhs, "after finer level");
    }
#else
    // final sol update (at level 0)  =
    //                   = solupdate[0] + P_0 * (solupdate[1] + P_1 * ( ...) )

    for (int level = num_levels - 1; level > 0; --level)
    {
        // tempvec[level-1] = P[level-1] * solupdate[level]
        P_Func[level-1]->Mult(*((*solupdate_lvls)[level]), *((*tempvec_lvls)[level - 1]));

        // solupdate[level-1] = solupdate[level-1] + P[level-1] * solupdate[level]
        for ( int blk = 0; blk < numblocks; ++blk)
            (*solupdate_lvls)[level - 1]->GetBlock(blk) += (*tempvec_lvls)[level - 1]->GetBlock(blk);
    }

    // 4. update the global iterate by the computed update (interpolated to the finest level)
    next_sol += *((*solupdate_lvls)[0]);
#endif

#ifdef DEBUG_INFO
    CheckConstrRes(next_sol.GetBlock(0), Constr, ConstrRhs, "after all levels update");
#endif

    // 5. restore sizes of righthand side vectors for the constraint
    // which were changed during transfer between levels
    rhs_constr->SetSize(ConstrRhs.Size());
    Qlminus1_f->SetSize(rhs_constr->Size());

    // for all but 1st iterate the rhs in the constraint will be 0
    // FIXME: this is duplicating the block in the beginning of the iteration,
    // the latter can be eliminated after debugging
    if (*current_iterate == 0)
    {
        *rhs_constr = 0.0;
    }

    ++(*current_iterate);

    return;
}

void BaseGeneralMinConstrSolver::ProjectFinerL2ToCoarser(int level, const Vector& in, Vector& ProjTin, Vector &out) const
{
    const SparseMatrix * Proj = P_L2[level];

    ProjTin.SetSize(Proj->Width());
    Proj->MultTranspose(in, ProjTin);

    const SparseMatrix * AE_e_lvl = AE_e[level];
    for ( int i = 0; i < ProjTin.Size(); ++i)
        ProjTin[i] /= AE_e_lvl->RowSize(i) * 1.;

    out.SetSize(Proj->Height());
    Proj->Mult(ProjTin, out);

    // FIXME: We need either to find additional memory for storing
    // result of the previous division in a temporary vector or
    // to multiply the output (ProjTin) back as in the loop below
    // in order to get correct output ProjTin in the end
    for ( int i = 0; i < ProjTin.Size(); ++i)
        ProjTin[i] *= AE_e_lvl->RowSize(i);

    return;
}

// start_level and end_level must be in 0-based indexing
// (*) uses tempvec_lvls for storing intermediate results
void BaseGeneralMinConstrSolver::InterpolateBack(int start_level, BlockVector& vec_start, int end_level, BlockVector& vec_end) const
{
    MFEM_ASSERT(start_level > end_level, "Interpolation makes sense only to the finer levels \n");

    *((*tempvec_lvls)[start_level]) = vec_start;

    for (int lvl = start_level; lvl > end_level; --lvl)
    {
        P_Func[lvl-1]->Mult(*((*tempvec_lvls)[lvl]), *((*tempvec_lvls)[lvl-1]));
    }

    vec_end = *((*tempvec_lvls)[end_level]);

    return;
}


// Righthand side at level l is of the form:
//   rhs_l = (Q_l - Q_{l+1}) where Q_k is an orthogonal L2-projector: W -> W_k
// or, equivalently,
//   rhs_l = (I - Q_{l-1,l}) rhs_{l-1},
// where Q_{k,k+1} is an orthogonal L2-projector W_{k+1} -> W_k,
// and rhs_{l-1} = Q_{l-1} f (setting Q_0 = Id)
// Hence,
//   Q_{l-1,l} = P_l * inv(P_l^T P_l) * P_l^T
// where P_l columns compose the basis of the coarser space.
// (*) Uses workfvec as an intermediate buffer
void BaseGeneralMinConstrSolver::ComputeLocalRhsConstr(int level) const
{
    // 1. rhs_constr = Q_{l-1,l} * Q_{l-1} * f = Q_l * f
    //    workfvec = P_l^T * Q_{l-1} * f
    ProjectFinerL2ToCoarser(level, *Qlminus1_f, *workfvec, *rhs_constr);

    // 2. rhs_constr = Q_l f - Q_{l-1}f
    *rhs_constr -= *Qlminus1_f;

    // 3. rhs_constr (new) = - rhs_constr(old) = Q_{l-1} f - Q_l f
    *rhs_constr *= -1;

    // 3. Q_{l-1} (new) = P_L2T[level] * f
    *Qlminus1_f = *workfvec;

    return;
}

// Computes prerequisites required for solving local problems at level l
// such as relation tables between AEs and internal fine-grid dofs
// and maybe smth else ... ?
void BaseGeneralMinConstrSolver::SetUpFinerLvl(int lvl) const
{
    (*AE_edofs_L2)[lvl] = mfem::Mult(*(AE_e[lvl]), *(el_to_dofs_L2[lvl]));
    (*AE_eintdofs_Func)[lvl] = Get_AE_eintdofs(lvl, *el_to_dofs_Func[lvl], bdrdofs_Func);

    /*
#ifdef COMPARE_WITH_OLD
    {
        ofstream ofs("newsolver_out_M_fine.txt");
        Funct.GetBlock(0,0).Print(ofs,1);
    }
#endif
    */

    // Funct_lvls[lvl] stores the Functional matrix on level lvl + 1
    BlockMatrix * Funct_PR;
    BlockMatrix * P_FuncT = Transpose(*P_Func[lvl]);
    if (lvl == 0)
        Funct_PR = mfem::Mult(Funct,*P_Func[lvl]);
    else
        Funct_PR = mfem::Mult(*(*Funct_lvls)[lvl - 1],*P_Func[lvl]);
    (*Funct_lvls)[lvl] = mfem::Mult(*P_FuncT, *Funct_PR);

    SparseMatrix *P_L2T = Transpose(*P_L2[lvl]);
    SparseMatrix *Constr_PR;
    if (lvl == 0)
        Constr_PR = mfem::Mult(Constr, P_Func[lvl]->GetBlock(0,0));
    else
        Constr_PR = mfem::Mult(*(*Constr_lvls)[lvl - 1], P_Func[lvl]->GetBlock(0,0));
    (*Constr_lvls)[lvl] = mfem::Mult(*P_L2T, *Constr_PR);

    (*tempvec_lvls)[lvl + 1] = new BlockVector((*Funct_lvls)[lvl]->RowOffsets());
    (*solupdate_lvls)[lvl + 1] = new BlockVector((*Funct_lvls)[lvl]->RowOffsets());
    (*rhsfunc_lvls)[lvl + 1] = new BlockVector((*Funct_lvls)[lvl]->RowOffsets());

    delete Funct_PR;
    delete Constr_PR;
    delete P_FuncT;
    delete P_L2T;

    return;
}

// Returns a pointer to a SparseMatrix which stores
// the relation between agglomerated elements (AEs)
// and fine-grid internal (w.r.t. to AEs) dofs.
// For lowest-order elements all the fine-grid dofs will be
// located at the boundary of fine-grid elements and not inside, but
// for higher order elements there will be two parts,
// one for dofs at fine-grid element faces which belong to the global boundary
// and a different treatment for internal (w.r.t. to fine elements) dofs
BlockMatrix* BaseGeneralMinConstrSolver::Get_AE_eintdofs(int level,
                            BlockMatrix& el_to_dofs, const std::vector<Array<Array<int>*> > &dof_is_bdr) const
{
    SparseMatrix * TempSpMat = new SparseMatrix;

    Array<int> res_rowoffsets(numblocks+1);
    res_rowoffsets[0] = 0;
    for (int blk = 0; blk < numblocks; ++blk)
        res_rowoffsets[blk + 1] = res_rowoffsets[blk] + AE_e[level]->Height();
    Array<int> res_coloffsets(numblocks+1);
    res_coloffsets[0] = 0;
    for (int blk = 0; blk < numblocks; ++blk)
    {
        *TempSpMat = el_to_dofs.GetBlock(blk,blk);
        res_coloffsets[blk + 1] = res_coloffsets[blk] + TempSpMat->Width();
    }

    BlockMatrix * res = new BlockMatrix(res_rowoffsets, res_coloffsets);

    Array<int> TempBdrDofs;
    for (int blk = 0; blk < numblocks; ++blk)
    {
        *TempSpMat = el_to_dofs.GetBlock(blk,blk);
        TempBdrDofs.MakeRef(*dof_is_bdr[blk][level]);

        // creating dofs_to_AE relation table
        SparseMatrix * dofs_AE = Transpose(*mfem::Mult(*AE_e[level], *TempSpMat));
        int ndofs = dofs_AE->Height();

        int * dofs_AE_i = dofs_AE->GetI();
        int * dofs_AE_j = dofs_AE->GetJ();
        double * dofs_AE_data = dofs_AE->GetData();

        int * innerdofs_AE_i = new int [ndofs + 1];

        // computing the number of internal degrees of freedom in all AEs
        int nnz = 0;
        for (int dof = 0; dof < ndofs; ++dof)
        {
            innerdofs_AE_i[dof]= nnz;
            for (int j = dofs_AE_i[dof]; j < dofs_AE_i[dof+1]; ++j)
            {
                // if a dof belongs to only one fine-grid element and is not at the domain boundary
                bool inside_finegrid_el = (higher_order && !TempBdrDofs[dof] && dofs_AE_data[j] == 1);
                MFEM_ASSERT( ( !inside_finegrid_el || (dofs_AE_i[dof+1] - dofs_AE_i[dof] == 1) ),
                        "A fine-grid dof inside a fine-grid element cannot belong to more than one AE");
                // if a dof is shared by two fine grid elements inside a single AE
                // OR a dof is strictly internal to a fine-grid element,
                // then it is an internal dof for this AE
                if (dofs_AE_data[j] == 2 || inside_finegrid_el )
                    nnz++;
            }

        }
        innerdofs_AE_i[ndofs] = nnz;

        // allocating j and data arrays for the created relation table
        int * innerdofs_AE_j = new int[nnz];
        double * innerdofs_AE_data = new double[nnz];

        int nnz_count = 0;
        for (int dof = 0; dof < ndofs; ++dof)
            for (int j = dofs_AE_i[dof]; j < dofs_AE_i[dof+1]; ++j)
            {
                bool inside_finegrid_el = (higher_order && !TempBdrDofs[dof] && dofs_AE_data[j] == 1);
                if (dofs_AE_data[j] == 2 || inside_finegrid_el )
                    innerdofs_AE_j[nnz_count++] = dofs_AE_j[j];
            }

        std::fill_n(innerdofs_AE_data, nnz, 1);

        // creating a relation between internal fine-grid dofs (w.r.t to AE) and AEs,
        // keeeping zero rows for non-internal dofs
        SparseMatrix * innerdofs_AE = new SparseMatrix(innerdofs_AE_i, innerdofs_AE_j, innerdofs_AE_data,
                                                       dofs_AE->Height(), dofs_AE->Width());

        delete dofs_AE;

        res->SetBlock(blk, blk, Transpose(*innerdofs_AE));
        //return Transpose(*innerdofs_AE);
    }

    return res;
}

// MFEM doesn't have a const version of GetSubMatrix
void GetSubMatrix(const SparseMatrix& SpMat, const Array<int> &rows, const Array<int> &cols,
                                DenseMatrix &subm)
{
   int i, j, gi, gj, s, t;
   double a;

   for (i = 0; i < rows.Size(); i++)
   {
      if ((gi = rows[i]) < 0) { gi = -1-gi, s = -1; }
      else { s = 1; }
      MFEM_ASSERT(gi < SpMat.Height(),
                  "Trying to insert a row " << gi << " outside the matrix height "
                  << SpMat.Height());
      SpMat.SetColPtr(gi);
      for (j = 0; j < cols.Size(); j++)
      {
         if ((gj = cols[j]) < 0) { gj = -1-gj, t = -s; }
         else { t = s; }
         MFEM_ASSERT(gj < SpMat.Width(),
                     "Trying to insert a column " << gj << " outside the matrix width "
                     << SpMat.Width());
         a = SpMat._Get_(gj);
         subm(i, j) = (t < 0) ? (-a) : (a);
      }
      SpMat.ClearColPtr();
   }
}


void BaseGeneralMinConstrSolver::SolveLocalProblems(int level, BlockVector& lvlrhs_func, Vector& rhs_constr, BlockVector& sol_update) const
{
    // FIXME: factorization can be done only during the first solver iterate, then stored and re-used
    //DenseMatrix sub_A;
    DenseMatrix sub_B;
    Vector sub_Constr;
    Array<int> sub_Func_offsets(numblocks + 1);

    //SparseMatrix A_fine = Funct.GetBlock(0,0);
    //SparseMatrix B_fine = Constr;
    const SparseMatrix * Constr_fine;
    if (level == 0)
        Constr_fine = &Constr;
    else
        Constr_fine = (*Constr_lvls)[level - 1];

    const BlockMatrix * Funct_fine;
    if (level == 0)
        Funct_fine = &Funct;
    else
        Funct_fine = (*Funct_lvls)[level - 1];

    *((*tempvec_lvls)[level]) = 0.0;

    // loop over all AE, solving a local problem in each AE
    std::vector<DenseMatrix> LocalAE_Matrices(numblocks);
    std::vector<Array<int>*> Local_inds(numblocks);
    int nAE = (*AE_edofs_L2)[level]->Height();
    bool is_degenerate = true;
    for( int AE = 0; AE < nAE; ++AE)
    {
        //std::cout << "AE = " << AE << "\n";
        sub_Func_offsets[0] = 0;
        for ( int blk = 0; blk < numblocks; ++blk )
        {
            //std::cout << "blk = " << blk << "\n";
            SparseMatrix FunctBlk = Funct_fine->GetBlock(blk,blk);
            SparseMatrix AE_eintdofsBlk = (*AE_eintdofs_Func)[level]->GetBlock(blk,blk);
            Array<int> tempview_inds(AE_eintdofsBlk.GetRowColumns(AE), AE_eintdofsBlk.RowSize(AE));
            //tempview_inds.Print();
            Local_inds[blk] = new Array<int>;
            tempview_inds.Copy(*Local_inds[blk]);

            sub_Func_offsets[blk + 1] = sub_Func_offsets[blk] + Local_inds[blk]->Size();

            if (blk == 0) // sigma block
            {
                Array<int> Wtmp_j((*AE_edofs_L2)[level]->GetRowColumns(AE), (*AE_edofs_L2)[level]->RowSize(AE));
                sub_B.SetSize(Wtmp_j.Size(), Local_inds[blk]->Size());
                GetSubMatrix(*Constr_fine, Wtmp_j, *Local_inds[blk], sub_B);

                if (*current_iterate == 0)
                    rhs_constr.GetSubVector(Wtmp_j, sub_Constr);
                else
                {
                    sub_Constr.SetSize(Wtmp_j.Size());
                    sub_Constr = 0.0;
                }

                //std::cout << "Local_inds[0]->Size = " << Local_inds[0]->Size() << "\n";

                /*
                if (level == 0)
                {
                    for (int i = 0; i < Local_inds[blk]->Size(); ++i)
                    {
                        if ( (*(ess_dof_finestlvl[0]))[(*Local_inds[blk])[i]] != 0 )
                        {
                            is_degenerate = true;
                            sub_B.SetCol(i,0.0);
                            sub_F[i] = (*(ess_dofvalues_finestlvl[0]))[(*Local_inds[blk])[i]];
                        }
                    }
                }
                */
            } // end of special treatment of the first block involved into constraint

            // Setting size of Dense Matrices
            LocalAE_Matrices[blk].SetSize(Local_inds[blk]->Size());

            // Obtaining submatrices:
            FunctBlk.GetSubMatrix(*Local_inds[blk], *Local_inds[blk], LocalAE_Matrices[blk]);

            /*
            if (level == 0)
            {
                for (int i = 0; i < Local_inds[blk]->Size(); ++i)
                {
                    if ( (*(ess_dof_finestlvl[blk]))[(*Local_inds[blk])[i]] != 0 )
                    {
                        LocalAE_Matrices[blk].SetRow(i,0.0);
                        LocalAE_Matrices[blk](i,i) = 1.0;
                    }
                }
            }
            */
        } // end of loop over all blocks

        BlockVector sub_Func(sub_Func_offsets);

        for ( int blk = 0; blk < numblocks; ++blk )
        {
            lvlrhs_func.GetBlock(blk).GetSubVector(*Local_inds[blk], sub_Func.GetBlock(blk));

            /*
            // FIXME: do we need this here?
            for (int i = 0; i < Local_inds[blk]->Size(); ++i)
            {
                if ( (*(bdrdofs_Func[blk][level]))[(*Local_inds[blk])[i]] != 0 )
                {
                    sub_Func.GetBlock(blk)[i] = 0.0;
                }
            }
            */
            //if (level == 0)
                //sub_Func.GetBlock(blk) = 0.0;
        }

        //MFEM_ASSERT(sub_Constr.Sum() < 1.0e-13, "checking local average at each level " << sub_Constr.Sum());

        BlockVector sol_loc(sub_Func_offsets);
        sol_loc = 0.0;

        /*
        if (level == 1 && AE == 0)
        {
            std::cout << "AE = 0, level = 1 \n";
            std::cout << "sub_M \n";
            LocalAE_Matrices[0].Print();
            std::cout << "sub_B \n";
            sub_B.Print();
            std::cout << "sub_Func \n";
            sub_Func.GetBlock(0).Print();
            std::cout << "sub_Constr \n";
            sub_Constr.Print();
        }
        */

        // Solving local problem at the agglomerate element AE:
        SolveLocalProblem(LocalAE_Matrices, sub_B, sub_Func, sub_Constr, sol_loc, is_degenerate);

        /*
        if (level == 1 && AE == 0)
        {
            std::cout << "sol_loc \n";
            sol_loc.GetBlock(0).Print();
            std::cout << "sub_B * sol_loc[0] \n";
            Vector temp(sub_B.Height());
            sub_B.Mult(sol_loc.GetBlock(0), temp);
            temp.Print();
            std::cout << "? \n \n";
        }
        */

        // computing solution as a vector at current level
        for ( int blk = 0; blk < numblocks; ++blk )
        {
            sol_update.GetBlock(blk).AddElementVector
                    (*Local_inds[blk], sol_loc.GetBlock(blk));
        }

    } // end of loop over AEs

    return;
}

// Sets up the coarse problem: matrix and righthand side
void BaseGeneralMinConstrSolver::SetUpCoarsestLvl() const
{
    // 1. eliminating boundary conditions at coarse level
    //bdrdofs_R[0][1]->Print();
    //(*Constr_lvls)[num_levels-1-1]->Print();

    /*
#ifdef COMPARE_WITH_OLD
    std::cout << "Looking at B_coarse in div part before eliminating bdr conds \n";
    (*Constr_lvls)[num_levels-1-1]->Print();
    //std::cout << "Looking at bdr dofs \n";
    //for ( int i = 0; i < ess_dof_coarselvl.Size(); ++i )
        //std::cout << ess_dof_coarselvl[i] << "\n";
#endif
    */

    /*
    // FIXME: get rid of temp, cannot use ess_dof_coarselvl directly
    // because of non-const version of EliminateCols
    Array<int> temp;
    // FIXME: should be several Array<int>'s for different components
    ess_dof_coarselvl[0]->Copy(temp);
    (*Constr_lvls)[num_levels-1-1]->EliminateCols(temp);

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        for ( int dof = 0; dof < temp.Size(); ++dof)
            if (temp[dof] != 0)
                (*Funct_lvls)[num_levels-1-1]->GetBlock(blk,blk).EliminateRowCol(dof);
    }
    */

    Array<int> temp;
    bdrdofs_Func[0][num_levels-1]->Copy(temp);
    //ess_dof_coarsest.Copy(temp);

    (*Constr_lvls)[num_levels-1-1]->EliminateCols(temp);

#ifdef COMPARE_WITH_OLD
    {
        ofstream ofs("newsolver_out_M_coarse_beforebnd.txt");
        (*Funct_lvls)[num_levels-1-1]->GetBlock(0,0).Print(ofs,1);
    }
#endif

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        Array<int> temp;
        bdrdofs_Func[blk][num_levels-1]->Copy(temp);
        //ess_dof_coarsest.Copy(temp);
        for ( int dof = 0; dof < temp.Size(); ++dof)
            if (temp[dof] != 0)
            {
                (*Funct_lvls)[num_levels-1-1]->GetBlock(blk,blk).EliminateRowCol(dof);
                (*rhsfunc_lvls)[num_levels-1]->GetBlock(blk)[dof] = 0.0;
            }
    }

    // Getting rid of the one-dimensional kernel for lambda in the coarsest level problem
    (*Constr_lvls)[num_levels-1-1]->EliminateRow(0);
    (*Constr_lvls)[num_levels-1-1]->GetData()[0] = 1.0;
    (*rhs_constr)[0] = 0.0;


#ifdef COMPARE_WITH_OLD
    {
        ofstream ofs("newsolver_out_M_coarse.txt");
        (*Funct_lvls)[num_levels-1-1]->GetBlock(0,0).Print(ofs,1);
    }
#endif

#ifdef COMPARE_WITH_OLD
    {
        ofstream ofs("newsolver_out_B_coarse.txt");
        (*Constr_lvls)[num_levels-1-1]->Print(ofs,1);
    }
#endif

    // 2. Creating the block matrix from the local parts using dof_truedof relation

    HypreParMatrix * Constr_global = dof_trueDof_Func[0]->LeftDiagMult(
                *((*Constr_lvls)[num_levels-1-1]), dof_trueDof_L2.GetColStarts());

    HypreParMatrix *ConstrT_global = Constr_global->Transpose();

    std::vector<HypreParMatrix*> Funct_d_td(numblocks);
    std::vector<HypreParMatrix*> d_td_T(numblocks);
    std::vector<HypreParMatrix*> Funct_global(numblocks);
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        Funct_d_td[blk] = dof_trueDof_Func[blk]->LeftDiagMult((*Funct_lvls)[num_levels-1-1]->GetBlock(blk,blk));
        d_td_T[blk] = dof_trueDof_Func[blk]->Transpose();

        Funct_global[blk] = ParMult(d_td_T[blk], Funct_d_td[blk]);
    }

    coarse_offsets = new Array<int>(numblocks + 2);
    (*coarse_offsets)[0] = 0;
    for ( int blk = 0; blk < numblocks; ++blk)
        (*coarse_offsets)[blk + 1] = Funct_global[blk]->Height();
    (*coarse_offsets)[numblocks + 1] = Constr_global->Height();
    (*coarse_offsets).PartialSum();


    coarse_matrix = new BlockOperator(*coarse_offsets);
    for ( int blk = 0; blk < numblocks; ++blk)
        coarse_matrix->SetBlock(blk, blk, Funct_global[blk]);
    coarse_matrix->SetBlock(0, numblocks, ConstrT_global);
    coarse_matrix->SetBlock(numblocks, 0, Constr_global);

    /*
#ifdef COMPARE_WITH_OLD
    SparseMatrix M_global_sp;
    Funct_global[0]->GetDiag(M_global_sp);
    std::cout << "M_global \n";
    M_global_sp.Print();
    //SparseMatrix B_global_sp;
    //Constr_global->GetDiag(B_global_sp);
    //std::cout << "B_global \n";
    //B_global_sp.Print();
#endif
    */

    // preconditioner for the coarse problem

    std::vector<Operator*> Funct_prec(numblocks);
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        Funct_prec[blk] = new HypreDiagScale(*Funct_global[blk]);
        ((HypreDiagScale*)(Funct_prec[blk]))->iterative_mode = false;
    }

    HypreParMatrix *MinvBt = Constr_global->Transpose();
    HypreParVector *Md = new HypreParVector(MPI_COMM_WORLD, Funct_global[0]->GetGlobalNumRows(),
                                            Funct_global[0]->GetRowStarts());
    Funct_global[0]->GetDiag(*Md);
    MinvBt->InvScaleRows(*Md);
    HypreParMatrix *Schur = ParMult(Constr_global, MinvBt);

    HypreBoomerAMG * invSchur = new HypreBoomerAMG(*Schur);
    invSchur->SetPrintLevel(0);
    invSchur->iterative_mode = false;

    coarse_prec = new BlockDiagonalPreconditioner(*coarse_offsets);
    for ( int blk = 0; blk < numblocks; ++blk)
        coarse_prec->SetDiagonalBlock(0, Funct_prec[blk]);
    coarse_prec->SetDiagonalBlock(numblocks, invSchur);
}


void BaseGeneralMinConstrSolver::SolveCoarseProblem(BlockVector& coarserhs_func, Vector& rhs_constr, BlockVector& sol_coarse) const
{
    std::cout << "SolveCoarseProblem is not implemented in the base class! \n";
    return;
}

class MinConstrSolver : private BaseGeneralMinConstrSolver
{
protected:
    virtual void SolveLocalProblem(std::vector<DenseMatrix> &FunctBlks, DenseMatrix& B,
                                   BlockVector &G, Vector& F, BlockVector &sol, bool is_degenerate) const;
    virtual void SolveCoarseProblem(BlockVector& coarserhs_func, Vector& coarserhs_constr, BlockVector& sol_coarse) const;
public:
    // constructors
    MinConstrSolver(int NumLevels, const Array< SparseMatrix*> &AE_to_e,
                           const Array< BlockMatrix*> &El_to_dofs_Func, const Array< SparseMatrix*> &El_to_dofs_L2,
                           const std::vector<HypreParMatrix*>& Dof_TrueDof_Func,
                           const HypreParMatrix& Dof_TrueDof_L2,
                           const Array< BlockMatrix*> &Proj_Func, const Array< SparseMatrix*> &Proj_L2,
                           const std::vector<Array<Array<int> *> > &BdrDofs_Func,
                           const BlockMatrix& FunctBlockMat,
                           const SparseMatrix& ConstrMat, const Vector& ConstrRhsVec,
                           const BlockVector& Bdrdata_Finest,
                           const Array<int>& Ess_dof_coarsest,
                           bool Higher_Order_Elements = false):
        BaseGeneralMinConstrSolver(NumLevels, AE_to_e, El_to_dofs_Func, El_to_dofs_L2,
                                   Dof_TrueDof_Func, Dof_TrueDof_L2,  Proj_Func, Proj_L2, BdrDofs_Func,
                                   FunctBlockMat, ConstrMat, ConstrRhsVec,
                                   Bdrdata_Finest, Ess_dof_coarsest,
                                   Higher_Order_Elements)
        {
            MFEM_ASSERT(numblocks == 1, "MinConstrSolver is designed for the formulation with"
                                    " sigma only but more blocks were provided!");
        }

    virtual void Mult(const Vector & x, Vector & y) const
    { BaseGeneralMinConstrSolver::Mult(x,y); }

};

// Solves a local linear system of the form
// [ A  BT ] [ sig ] = [ G ]
// [ B  0  ] [ lam ] = [ F ]
// as
// lambda = inv (BinvABT) * ( B * invA * G - F )
// sig = invA * (G - BT * lambda) = invA * G - invA * BT * lambda
void MinConstrSolver::SolveLocalProblem(std::vector<DenseMatrix> &FunctBlks, DenseMatrix& B,
                                        BlockVector &G, Vector& F, BlockVector &sol, bool is_degenerate) const
{
    // FIXME: rewrite the routine

    // creating a Schur complement matrix Binv(A)BT
    //std::cout << "Inverting A: \n";
    //FunctBlks[0].Print();
    DenseMatrixInverse inv_A(FunctBlks[0]);

    // invAG = invA * G
    Vector invAG;
    inv_A.Mult(G, invAG);

    DenseMatrix BT(B.Width(), B.Height());
    BT.Transpose(B);

    DenseMatrix invABT;
    inv_A.Mult(BT, invABT);

    // Schur = BinvABT
    DenseMatrix Schur(B.Height(), invABT.Width());
    mfem::Mult(B, invABT, Schur);

    //std::cout << "Inverting Schur: \n";

    // getting rid of the one-dimensional kernel which exists for lambda
    //double tempF0;
    if (is_degenerate)
    {
        Schur.SetRow(0,0);
        Schur.SetCol(0,0);
        Schur(0,0) = 1.;
        //tempF0 = F(0);
        //F(0) = 0;
    }

    //Schur.Print();
    DenseMatrixInverse inv_Schur(Schur);

    // temp = ( B * invA * G - F )
    Vector temp(B.Height());
    B.Mult(invAG, temp);
    if (*current_iterate == 0) // else it is simply 0
    {
        //F(0) = tempF0;
        temp -= F;
    }

    if (is_degenerate)
    {
        temp(0) = 0;
    }

    // lambda = inv(BinvABT) * ( B * invA * G - F )
    Vector lambda(inv_Schur.Height());
    inv_Schur.Mult(temp, lambda);

    // temp2 = (G - BT * lambda)
    Vector temp2(B.Width());
    B.MultTranspose(lambda,temp2);
    temp2 *= -1;
    temp2 += G;

    // sig = invA * temp2 = invA * (G - BT * lambda)
    inv_A.Mult(temp2, sol.GetBlock(0));

    /*
    std::cout << "Check for the second equation: \n";
    Vector Ax(B.Height());
    B.Mult(sol.GetBlock(0), Ax);
    Ax -= F;
    Ax.Print();
    */

    return;
}

void MinConstrSolver::SolveCoarseProblem(BlockVector& coarserhs_func, Vector& coarserhs_constr, BlockVector& sol_coarse) const
{
    // 1. set up the solver parameters

    int maxIter(50000);
    double rtol(1.e-16);
    double atol(1.e-16);

    MINRESSolver solver(MPI_COMM_WORLD);
    solver.SetAbsTol(atol);
    solver.SetRelTol(rtol);
    solver.SetMaxIter(maxIter);
    solver.SetOperator(*coarse_matrix);
//#ifdef COMPARE_WITH_OLD
//    solver.SetPrintLevel(1);
//#else
    solver.SetPreconditioner(*coarse_prec);
    solver.SetPrintLevel(0);
//#endif

    // 2. set up solution and righthand side vectors
    BlockVector trueX(*coarse_offsets), trueRhs(*coarse_offsets);
    trueX = 0.0;

    //for ( int blk = 0; blk < coarse_offsets->Size() - 1; ++blk)
        //trueRhs.GetBlock(blk) = 0.0;
    trueRhs = 0.0;

    //std::cout << "size before " << trueRhs.GetBlock(0).Size() << "\n";
    MFEM_ASSERT(trueRhs.GetBlock(0).Size() == coarserhs_func.GetBlock(0).Size() &&
                trueRhs.GetBlock(1).Size() == coarserhs_constr.Size(),
                "Sizes mismatch when finalizing rhs at the coarsest level!\n");
    trueRhs.GetBlock(0) = coarserhs_func.GetBlock(0);
    //std::cout << "size after " << trueRhs.GetBlock(0).Size() << "\n";
    if (*current_iterate == 0) // else it is simply 0
        trueRhs.GetBlock(1) = coarserhs_constr;

    //trueRhs.GetBlock(0) = 0.0;

    /*
#ifdef COMPARE_WITH_OLD
    std::cout << "Looking at coarse rhs of size " << trueRhs.Size() << "\n";
    //std::cout << "trueRhs block 0 of size " << trueRhs.GetBlock(0).Size() << "\n";
    //trueRhs.GetBlock(0).Print();
    //std::cout << "rhs_constr inside SolveCoarseProblem (of size " << rhs_constr.Size() << ") \n";
    //rhs_constr.Print();
    std::cout << "trueRhs block 1 of size " << trueRhs.GetBlock(1).Size() << "\n";
    trueRhs.GetBlock(1).Print();
#endif
    */

    // 3. solve the linear system with preconditioned MINRES.
    solver.Mult(trueRhs, trueX);

    //std::cout << "trueRhs block 1 of size " << trueRhs.GetBlock(1).Size() << "\n";
    //trueRhs.GetBlock(1).Print();

    /*
#ifdef DEBUG_INFO
    std::cout << "Checking residual in the constraint after solving the coarsest level problem:\n";
    BlockVector res(*coarse_offsets);
    coarse_matrix->Mult(trueX, res);
    //ofstream ofs("newsolver_out_wrong.txt");
    //res.GetBlock(1).Print(ofs,1);
    res -= trueRhs;
    double constr_resnorm = res.GetBlock(1).Norml2() /
            sqrt (res.GetBlock(1).Size());
    std::cout << "constr_resnorm at the coarsest level = " << constr_resnorm << "\n";
#endif
    */

    // 4. convert solution from truedof to dof

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        dof_trueDof_Func[blk]->Mult(trueX.GetBlock(blk), sol_coarse.GetBlock(blk));
        //dof_trueDof_Func[blk]->Mult(trueX.GetBlock(blk), ((*tempvec_lvls)[num_levels-1])->GetBlock(blk));
        //(*tempvec_lvls)[num_levels-1]->GetBlock(blk) = trueX.GetBlock(blk);
    }

    return;
}

#if 0
class MinConstrSolverWithS : private BaseGeneralMinConstrSolver
{
private:
    const int strategy;

protected:
    virtual void SolveLocalProblem (std::vector<DenseMatrix> &FunctBlks, DenseMatrix& B, BlockVector &G, Vector& F, BlockVector &sol) const;
    virtual void SolveCoarseProblem(BlockVector& rhs_func, Vector& rhs_constr, BlockVector& sol_coarse) const;
    virtual void ComputeRhsFunc(BlockVector &rhs_func, const Vector& x) const;
    virtual void SetUpFinerLvl(int level) const
    { BaseGeneralMinConstrSolver::SetUpFinerLvl(level);}
public:
    // constructor
    MinConstrSolverWithS(int NumLevels, const Array< SparseMatrix*> &AE_to_e,
                         const Array< BlockMatrix*> &El_to_dofs_Func, const Array< SparseMatrix*> &El_to_dofs_L2,
                         const std::vector<HypreParMatrix*>& Dof_TrueDof_Func,
                         const HypreParMatrix& Dof_TrueDof_L2,
                         const Array< BlockMatrix*> &Proj_Func, const Array< SparseMatrix*> &Proj_L2,
                         const std::vector<Array<Array<int> *> > &BdrDofs_Func,
                         const BlockMatrix& FunctBlockMat,
                         const SparseMatrix& ConstrMat, const Vector& ConstrRhsVec,
                         const BlockVector& Bdrdata_Finest,
                         bool Higher_Order_Elements = false, int Strategy = 0)
        : BaseGeneralMinConstrSolver(NumLevels, AE_to_e, El_to_dofs_Func, El_to_dofs_L2,
                         Dof_TrueDof_Func, Dof_TrueDof_L2, Proj_Func, Proj_L2, BdrDofs_Func,
                         FunctBlockMat, ConstrMat, ConstrRhsVec,
                         Bdrdata_Finest,
                         Higher_Order_Elements),
         strategy(Strategy)
         {}

    virtual void Mult(const Vector & x, Vector & y) const;
};

void MinConstrSolverWithS::Mult(const Vector & x, Vector & y) const
{
    std::cout << "Mult() for (sigma,S) formulation is not implemented! \n";
    y = x;
}

// Computes rhs coming from the last iterate sigma
// rhs_func = - A * x, where A is the matrix arising
// from the local minimization functional, and x is the
// minimzed variables (sigma or (sigma,S)).
void MinConstrSolverWithS::ComputeRhsFunc(BlockVector &rhs_func, const Vector& x) const
{
    // if we going to minimize only sigma
    if (strategy != 0)
    {
        xblock->Update(x.GetData(), block_offsets);
        Funct.GetBlock(0,0).Mult(xblock->GetBlock(0), rhs_func);
    }
    else
    {
        xblock->Update(x.GetData(), block_offsets);
        Funct.Mult(*xblock, rhs_func);
        *rhs_func *= -1;
    }
}

// Solves a local linear system of the form
// [ A  DT  BT ] [ sig ] = [ Gsig ]
// [ D  0   0  ] [  s  ] = [ GS   ]
// [ B  0   0  ] [ lam ] = [ F    ]
// as
// [s, lam]^T = inv ( [D B]^T invA [DT BT] ) * ( [D B]^T invA * Gsig - [GS F]^T )
// s = [s, lam]_1
// sig = invA * (Gsig - [DT BT] * [s, lam]^T)
void MinConstrSolverWithS::SolveLocalProblem (std::vector<DenseMatrix> &FunctBlks, DenseMatrix& B, BlockVector &G, Vector& F, BlockVector &sol) const
{
    std::cout << "MinConstrSolverWithS::SolveLocalProblem() is not implemented!";
    // FIXME: rewrite the routine

    /*

    Array<int> offsets(3);
    offsets[0] = 0;
    offsets[1] = GS.Size();
    offsets[2] = F.Size();
    offsets.PartialSum();

    BlockVector s_lam(offsets);

    BlockDenseMatrix D_B(offsets);
    D_B.SetBlock(0,0,D);
    D_B.SetBlock(1,0,B);

    DenseMatrixInverse inv_A(A);
    BlockDenseMatrix invA_D_B;
    inv_A.Mult(D_B, invA_D_B);

    BlockDenseMatrix Schur;
    Mult(D_B, inv_A_DT_BT, Schur);

    DenseBlockMatrixInverse inv_Schur(Schur);

    s = s_lam.GetBlock(0);

    // computing sig
    // temp2 = Gsig - [DT BT] * [s, lam]^T
    Vector temp2;
    D_B.MultTranspose(s_lam, temp2);
    temp2 *= -1;
    temp2 += Gsig;

    // sig = invA * temp2
    inv_A.Mult(temp2, sig);
    */

    return;
}

void MinConstrSolverWithS::SolveCoarseProblem(BlockVector& rhs_func, Vector& rhs_constr, BlockVector& sol_coarse) const
{
    std::cout << "SolveCoarseProblem is not implemented! \n";
    return;
}
#endif

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
                   Vector &sigma,
                   Array<int>& ess_dof_coarsestlvl_list
                   )
    {
//        StopWatch chrono;

//        Vector sol_p_c2f;
        Vector vec1;

        Vector rhs_l;
        Vector comp;
        Vector F_coarse;

        Vector total_sig(P_R[0]->Height());
        total_sig = .0;

//        chrono.Clear();
//        chrono.Start();
#ifdef DEBUG_INFO
        SparseMatrix * B_input = B_fine;
#endif


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
            {
                //std::cout << "Diag(m) = " << Diag(m) << "\n";
                invDiag(m) = comp(m)/Diag(m);
            }

            //std::cout << "Diag(100) = " << Diag(100);
            //std::cout << "Diag(200) = " << Diag(200);
            //std::cout << "Diag(300) = " << Diag(300);


            P_W[l]->Mult(invDiag,F_coarse);



            rhs_l -=F_coarse;

            MFEM_ASSERT(rhs_l.Sum()<= 9e-11,
                        "Average of rhs at each level is not zero: " << rhs_l.Sum());


            if (l> 0) {

                // 4. Creating matrices for the coarse problem:
                SparseMatrix *P_WT2 = Transpose(*P_W[l-1]);
                SparseMatrix *P_RT2;
                if (M_fine)
                    P_RT2 = Transpose(*P_R[l-1]);

                SparseMatrix *B_PR = Mult(*B_fine, *P_R[l-1]);
                B_fine = Mult(*P_WT2, *B_PR);

                if (M_fine)
                {
                    SparseMatrix *M_PR = Mult(*M_fine, *P_R[l-1]);
                    M_fine = Mult(*P_RT2, *M_PR);
                }
            }

            //5. Setting for the coarse problem
            DenseMatrix sub_M;
            DenseMatrix sub_B;
            DenseMatrix sub_BT;
//            DenseMatrix invBB;

            Vector sub_F;
            Vector sub_G;

            //Vector to Assamble the solution at level l
            Vector u_loc_vec(AE_W->Width());
            Vector p_loc_vec(AE_R->Width());

            u_loc_vec =0.0;
            p_loc_vec =0.0;

            /*
#ifdef COMPARE_WITH_OLD
            std::cout << "rhs at fine level (size " << rhs_l.Size() << ") \n";
            rhs_l.Print(std::cout,1);
#endif
            */

            for( int e = 0; e < AE_R->Height(); e++){

                Array<int> Rtmp_j(AE_R->GetRowColumns(e), AE_R->RowSize(e));
                Array<int> Wtmp_j(AE_W->GetRowColumns(e), AE_W->RowSize(e));

                // Setting size of Dense Matrices
                if (M_fine)
                    sub_M.SetSize(Rtmp_j.Size());
                sub_B.SetSize(Wtmp_j.Size(),Rtmp_j.Size());
                sub_BT.SetSize(Rtmp_j.Size(),Wtmp_j.Size());
//                sub_G.SetSize(Rtmp_j.Size());
//                sub_F.SetSize(Wtmp_j.Size());

                // Obtaining submatrices:
                if (M_fine)
                    M_fine->GetSubMatrix(Rtmp_j,Rtmp_j, sub_M);
                B_fine->GetSubMatrix(Wtmp_j,Rtmp_j, sub_B);
                sub_BT.Transpose(sub_B);

//                sub_G  = .0;
//                sub_F  = .0;

                rhs_l.GetSubVector(Wtmp_j, sub_F);


                Vector sig(Rtmp_j.Size());

                MFEM_ASSERT(sub_F.Sum()<= 9e-11,
                            "checking local average at each level " << sub_F.Sum());

#ifdef MFEM_DEBUG
                Vector sub_FF = sub_F;
#endif

                // Solving local problem:
                Local_problem(sub_M, sub_B, sub_G, sub_F,sig);

                /*
#ifdef DEBUG_INFO
                if (e == 0 && l == 1)
                {
                    std::cout << "Looking at one local problem in div part, e = " << e << "\n";
                    std::cout << "Wtmp_j \n";
                    Wtmp_j.Print();
                    std::cout << "Rtmp_j \n";
                    Rtmp_j.Print();
                    std::cout << "sub_F \n";
                    sub_F.Print();
                    //std::cout << "rhs_l(partly) \n";
                    //std::cout << rhs_l[0] << "\n";
                    //for ( int i = 375; i < 395; ++i)
                        //std::cout << rhs_l[i] << "\n";
                    std::cout << "sub_M \n";
                    sub_M.Print();
                    std::cout << "sub_B \n";
                    sub_B.Print();
                    std::cout << "sig \n";
                    sig.Print();
                }
#endif
                */

#ifdef MFEM_DEBUG
                // Checking if the local problems satisfy the condition
                Vector fcheck(Wtmp_j.Size());
                fcheck =.0;
                sub_B.Mult(sig, fcheck);
                fcheck-=sub_FF;
                MFEM_ASSERT(fcheck.Norml2()<= 9e-11,
                            "checking local residual norm at each level " << fcheck.Norml2());
#endif

                p_loc_vec.AddElementVector(Rtmp_j,sig);

            } // end of loop over all elements at level l

#ifdef MFEM_DEBUG
            Vector fcheck2(u_loc_vec.Size());
            fcheck2 = .0;
            B_fine->Mult(p_loc_vec, fcheck2);
            fcheck2-=rhs_l;
            MFEM_ASSERT(fcheck2.Norml2()<= 9e-11,
                        "checking global solution at each level " << fcheck2.Norml2());
#endif

            /*
#ifdef COMPARE_WITH_OLD
            if (l == 1)
            {
                std::cout << "p_loc_vec before projection (from finer levels), size " << p_loc_vec.Size() << ": \n";
                //total_sig.Print(std::cout,1);
                ofstream ofs("div_part_out.txt");
                p_loc_vec.Print(ofs,1);
            }
#endif
            */

            // Final Solution ==
            if (l>0){
                for (int k = l-1; k>=0; k--){

                    vec1.SetSize(P_R[k]->Height());
                    P_R[k]->Mult(p_loc_vec, vec1);
                    p_loc_vec = vec1;

                }
            }

            /*
#ifdef COMPARE_WITH_OLD
            if (l == 1)
            {
                std::cout << "p_loc_vec after projection (from finer levels), size " << p_loc_vec.Size() << ": \n";
                //total_sig.Print(std::cout,1);
                ofstream ofs("div_part_out.txt");
                p_loc_vec.Print(ofs,1);
            }
#endif
            */

#ifdef DEBUG_INFO
            CheckConstrRes(p_loc_vec, *B_input, F_fine, "for the fine level");
            if (l == 0)
            {
                ofstream ofs("divpart_out_sol_level_0.txt");
                p_loc_vec.Print(ofs,1);
            }
            if (l == 1)
            {
                ofstream ofs("divpart_out_sol_level_1.txt");
                p_loc_vec.Print(ofs,1);
            }
#endif


            total_sig +=p_loc_vec;

            MFEM_ASSERT(total_sig.Norml2()<= 9e+9,
                        "checking global solution added" << total_sig.Norml2());

            /*
#ifdef COMPARE_WITH_OLD
            if (l == 0)
            {
                std::cout << "solution total_sig(from finer levels), size " << total_sig.Size() << ": \n";
                //total_sig.Print(std::cout,1);
                ofstream ofs("div_part_out.txt");
                total_sig.Print(ofs,1);
            }
#endif
            */

#ifdef DEBUG_INFO
            CheckConstrRes(total_sig, *B_input, F_fine, "after finer level");
#endif
        } // end of loop over levels

        /*
#ifdef COMPARE_WITH_OLD
        std::cout << "solution total_sig(from finer levels), size " << total_sig.Size() << ": \n";
        //total_sig.Print(std::cout,1);
        //ofstream ofs("div_part_out.txt");
        //total_sig.Print(ofs,1);
#endif
        */

        // The coarse problem::

        SparseMatrix *M_coarse;
        SparseMatrix *B_coarse;
        Vector FF_coarse(P_W[ref_levels-1]->Width());

        rhs_l +=F_coarse;
        P_W[ref_levels-1]->MultTranspose(rhs_l, FF_coarse );

        SparseMatrix *P_WT2 = Transpose(*P_W[ref_levels-1]);
        SparseMatrix *P_RT2;
        if (M_fine)
            P_RT2 = Transpose(*P_R[ref_levels-1]);

        SparseMatrix *B_PR = Mult(*B_fine, *P_R[ref_levels-1]);
        B_coarse = Mult(*P_WT2, *B_PR);

        /*
#ifdef COMPARE_WITH_OLD
        std::cout << "Looking at B_coarse in div part before eliminating boundary conds \n";
        B_coarse->Print();
        //std::cout << "Looking at ess_dof_coarsestlvl_list \n";
        //ess_dof_coarsestlvl_list.Print(std::cout,1);
#endif
        */

        B_coarse->EliminateCols(ess_dof_coarsestlvl_list);

#ifdef COMPARE_WITH_OLD
        {
            ofstream ofs("divpart_out_B_coarse.txt");
            B_coarse->Print(ofs,1);
        }
#endif

        /*
#ifdef COMPARE_WITH_OLD
        std::cout << "Looking at B_coarse in div part before using d_td \n";
        B_coarse->Print();
#endif
        */
        if (M_fine)
        {
#ifdef COMPARE_WITH_OLD
            {
                ofstream ofs("divpart_out_M_fine.txt");
                M_fine->Print(ofs,1);
            }
#endif
            SparseMatrix *M_PR = Mult(*M_fine, *P_R[ref_levels-1]);

            M_coarse =  Mult(*P_RT2, *M_PR);

#ifdef COMPARE_WITH_OLD
            {
                ofstream ofs("divpart_out_M_coarse_beforebnd.txt");
                M_coarse->Print(ofs,1);
            }
#endif
            for ( int k = 0; k < ess_dof_coarsestlvl_list.Size(); ++k)
                if (ess_dof_coarsestlvl_list[k] !=0)
                    M_coarse->EliminateRowCol(k);
        }

#ifdef COMPARE_WITH_OLD
        {
            ofstream ofs("divpart_out_M_coarse.txt");
            M_coarse->Print(ofs,1);
        }
#endif

        Vector sig_c(B_coarse->Width());

        auto B_Global = d_td_coarse_R->LeftDiagMult(*B_coarse,d_td_coarse_W->GetColStarts());
        Vector Truesig_c(B_Global->Width());

        if (M_fine)
        {
            auto d_td_M = d_td_coarse_R->LeftDiagMult(*M_coarse);
            HypreParMatrix *d_td_T = d_td_coarse_R->Transpose();

            HypreParMatrix *M_Global = ParMult(d_td_T, d_td_M);
            HypreParMatrix *BT = B_Global->Transpose();

            /*
#ifdef COMPARE_WITH_OLD
            SparseMatrix M_global_sp;
            M_Global->GetDiag(M_global_sp);
            std::cout << "M_global in div-part \n";
            M_global_sp.Print();
            //SparseMatrix B_global_sp;
            //B_Global->GetDiag(B_global_sp);
            //std::cout << "B_global in div-part \n";
            //B_global_sp.Print();
#endif
            */

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

#ifdef DEBUG_INFO
            {
                ofstream ofs("div_part_constr_coarserhs.txt");
                FF_coarse.Print(ofs,1);
            }
#endif

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
            double rtol(1.e-16);
            double atol(1.e-16);

            MINRESSolver solver(MPI_COMM_WORLD);
            solver.SetAbsTol(atol);
            solver.SetRelTol(rtol);
            solver.SetMaxIter(maxIter);
            solver.SetOperator(coarseMatrix);
//#ifdef COMPARE_WITH_OLD
//            solver.SetPrintLevel(1);
//#else
            solver.SetPreconditioner(*darcyPr);
            solver.SetPrintLevel(0);
//#endif
            trueX = 0.0;
            solver.Mult(trueRhs, trueX);
//            chrono.Stop();

            /*
#ifdef COMPARE_WITH_OLD
            std::cout << "Looking at coarse rhs in divpart of size " << trueRhs.Size() << "\n";
            //std::cout << "trueRhs block 0 of size " << trueRhs.GetBlock(0).Size() << "\n";
            //trueRhs.GetBlock(0).Print();
            std::cout << "trueRhs block 1 of size " << trueRhs.GetBlock(1).Size() << "\n";
            trueRhs.GetBlock(1).Print(std::cout,1);
#endif
            */


//            cout << "MINRES converged in " << solver.GetNumIterations() << " iterations" <<endl;
//            cout << "MINRES solver took " << chrono.RealTime() << "s. \n";
            Truesig_c = trueX.GetBlock(0);
        }
        else
        {
            int maxIter(50000);
            double rtol(1.e-16);
            double atol(1.e-16);

            HypreParMatrix *MinvBt = B_Global->Transpose();
            HypreParMatrix *S = ParMult(B_Global, MinvBt);

            auto invS = new HypreBoomerAMG(*S);
            invS->SetPrintLevel(0);
            invS->iterative_mode = false;

            Vector tmp_c(B_Global->Height());
            tmp_c = 0.0;

            CGSolver solver(MPI_COMM_WORLD);
            solver.SetAbsTol(atol);
            solver.SetRelTol(rtol);
            solver.SetMaxIter(maxIter);
            solver.SetOperator(*S);
            solver.SetPreconditioner(*invS);
            solver.SetPrintLevel(0);
            solver.Mult(FF_coarse, tmp_c);
//            chrono.Stop();

//            cout << "CG converged in " << solver.GetNumIterations() << " iterations" <<endl;
//            cout << "CG solver took " << chrono.RealTime() << "s. \n";
            MinvBt->Mult(tmp_c, Truesig_c);
        }

        d_td_coarse_R->Mult(Truesig_c,sig_c);

#ifdef DEBUG_INFO
        {
            ofstream ofs("divpart_out_sol_level_coarse_at_coarse.txt");
            sig_c.Print(ofs,1);
        }
#endif

        for (int k = ref_levels-1; k>=0; k--){

            vec1.SetSize(P_R[k]->Height());
            P_R[k]->Mult(sig_c, vec1);
            sig_c.SetSize(P_R[k]->Height());
            sig_c = vec1;

        }
#ifdef DEBUG_INFO
        CheckConstrRes(sig_c, *B_input, F_fine, "for the coarsest level");
        {
            ofstream ofs("divpart_out_sol_level_coarse.txt");
            sig_c.Print(ofs,1);
        }
#endif

        total_sig+=sig_c;
        sigma.SetSize(total_sig.Size());
        sigma = total_sig;

#ifdef DEBUG_INFO
        CheckConstrRes(sigma, *B_input, F_fine, "after all levels update");
#endif

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


        DenseMatrix sub_BT(sub_B.Width(), sub_B.Height());
        sub_BT.Transpose(sub_B);

        DenseMatrix invM_BT;
        if (sub_M.Size() > 0)
        {
            DenseMatrixInverse invM_loc(sub_M);
            invM_loc.Mult(sub_BT,invM_BT);
        }

        /* Solving the local problem:
                  *
              * Msig + B^tu = G
              * Bsig        = F
              *
              * sig =  M^{-1} B^t(-u) + M^{-1} G
              *
              * B M^{-1} B^t (-u) = F
              */

        DenseMatrix B_invM_BT(sub_B.Height());

        if (sub_M.Size() > 0)
            Mult(sub_B, invM_BT, B_invM_BT);
        else
            Mult(sub_B, sub_BT, B_invM_BT);

//        Vector one(sub_B.Height());
//        one = 0.0;
//        one[0] =1;
        B_invM_BT.SetRow(0,0);
        B_invM_BT.SetCol(0,0);
//        B_invM_BT.SetCol(0,one);
        B_invM_BT(0,0)=1.;


        DenseMatrixInverse inv_BinvMBT(B_invM_BT);

//        Vector invMG(sub_M.Size());
//        invM_loc.Mult(Sub_G,invMG);

        sub_F[0] = 0;
        Vector uu(sub_B.Height());
        inv_BinvMBT.Mult(sub_F, uu);
        if (sub_M.Size() > 0)
            invM_BT.Mult(uu,sigma);
        else
            sub_BT.Mult(uu,sigma);
//        sigma += invMG;
    }

};

class MonolithicMultigrid : public Solver
{
private:
    class BlockSmoother : public BlockOperator
    {
    public:
        BlockSmoother(BlockOperator &Op)
            :
              BlockOperator(Op.RowOffsets()),
              A01((HypreParMatrix&)Op.GetBlock(0,1)),
              A10((HypreParMatrix&)Op.GetBlock(1,0)),
              offsets(Op.RowOffsets())
        {
            HypreParMatrix &A00 = (HypreParMatrix&)Op.GetBlock(0,0);
            HypreParMatrix &A11 = (HypreParMatrix&)Op.GetBlock(1,1);

            B00 = new HypreSmoother(A00);
            B11 = new HypreSmoother(A11);

            tmp01.SetSize(A00.Width());
            tmp02.SetSize(A00.Width());
            tmp1.SetSize(A11.Width());
        }

        virtual void Mult(const Vector & x, Vector & y) const
        {
            yblock.Update(y.GetData(), offsets);
            xblock.Update(x.GetData(), offsets);

            yblock.GetBlock(0) = 0.0;
            B00->Mult(xblock.GetBlock(0), yblock.GetBlock(0));

            tmp1 = xblock.GetBlock(1);
            A10.Mult(-1.0, yblock.GetBlock(0), 1.0, tmp1);
            B11->Mult(tmp1, yblock.GetBlock(1));
        }

        virtual void MultTranspose(const Vector & x, Vector & y) const
        {
            yblock.Update(y.GetData(), offsets);
            xblock.Update(x.GetData(), offsets);

            yblock.GetBlock(1) = 0.0;
            B11->Mult(xblock.GetBlock(1), yblock.GetBlock(1));

            tmp01 = xblock.GetBlock(0);
            A01.Mult(-1.0, yblock.GetBlock(1), 1.0, tmp01);
            B00->Mult(tmp01, yblock.GetBlock(0));
        }

        virtual void SetOperator(const Operator &op) { }

        ~BlockSmoother()
        {
            delete B00;
            delete B11;
            delete S;
        }

    private:
        HypreSmoother *B00;
        HypreSmoother *B11;
        HypreParMatrix &A01;
        HypreParMatrix &A10;
        HypreParMatrix *S;

        const Array<int> &offsets;
        mutable BlockVector xblock;
        mutable BlockVector yblock;
        mutable Vector tmp01;
        mutable Vector tmp02;
        mutable Vector tmp1;
    };

public:
    MonolithicMultigrid(BlockOperator &Operator,
                        const Array<BlockOperator*> &P,
                        Solver *CoarsePrec=NULL)
        :
          Solver(Operator.RowOffsets().Last()),
          P_(P),
          Operators_(P.Size()+1),
          Smoothers_(Operators_.Size()),
          current_level(Operators_.Size()-1),
          correction(Operators_.Size()),
          residual(Operators_.Size()),
          CoarseSolver(NULL),
          CoarsePrec_(CoarsePrec)
    {
        Operators_.Last() = &Operator;

        for (int l = Operators_.Size()-1; l >= 0; l--)
        {
            Array<int>& Offsets = Operators_[l]->RowOffsets();
            correction[l] = new Vector(Offsets.Last());
            residual[l] = new Vector(Offsets.Last());

            HypreParMatrix &A00 = (HypreParMatrix&)Operators_[l]->GetBlock(0,0);
            HypreParMatrix &A11 = (HypreParMatrix&)Operators_[l]->GetBlock(1,1);
            HypreParMatrix &A01 = (HypreParMatrix&)Operators_[l]->GetBlock(0,1);

            // Define smoothers
            Smoothers_[l] = new BlockSmoother(*Operators_[l]);

            // Define coarser level operators - two steps RAP (or P^T A P)
            if (l > 0)
            {
                HypreParMatrix& P0 = (HypreParMatrix&)P[l-1]->GetBlock(0,0);
                HypreParMatrix& P1 = (HypreParMatrix&)P[l-1]->GetBlock(1,1);

                unique_ptr<HypreParMatrix> P0T(P0.Transpose());
                unique_ptr<HypreParMatrix> P1T(P1.Transpose());

                unique_ptr<HypreParMatrix> A00P0( ParMult(&A00, &P0) );
                unique_ptr<HypreParMatrix> A11P1( ParMult(&A11, &P1) );
                unique_ptr<HypreParMatrix> A01P1( ParMult(&A01, &P1) );

                HypreParMatrix *A00_c(ParMult(P0T.get(), A00P0.get()));
                A00_c->CopyRowStarts();
                HypreParMatrix *A11_c(ParMult(P1T.get(), A11P1.get()));
                A11_c->CopyRowStarts();
                HypreParMatrix *A01_c(ParMult(P0T.get(), A01P1.get()));
                A01_c->CopyRowStarts();
                HypreParMatrix *A10_c(A01_c->Transpose());

                Operators_[l-1] = new BlockOperator(P[l-1]->ColOffsets());
                Operators_[l-1]->SetBlock(0, 0, A00_c);
                Operators_[l-1]->SetBlock(0, 1, A01_c);
                Operators_[l-1]->SetBlock(1, 0, A10_c);
                Operators_[l-1]->SetBlock(1, 1, A11_c);
                Operators_[l-1]->owns_blocks = 1;
            }
        }

        if (CoarsePrec)
        {
            CoarseSolver = new CGSolver(
                        ((HypreParMatrix&)Operator.GetBlock(0,0)).GetComm() );
            CoarseSolver->SetRelTol(1e-8);
            CoarseSolver->SetMaxIter(50);
            CoarseSolver->SetPrintLevel(0);
            CoarseSolver->SetOperator(*Operators_[0]);
            CoarseSolver->SetPreconditioner(*CoarsePrec);
        }
    }

    virtual void Mult(const Vector & x, Vector & y) const;

    virtual void SetOperator(const Operator &op) { }

    ~MonolithicMultigrid()
    {
        for (int l = 0; l < Operators_.Size(); l++)
        {
            delete Smoothers_[l];
            delete correction[l];
            delete residual[l];
        }
    }

private:
    void MG_Cycle() const;

    const Array<BlockOperator*> &P_;

    Array<BlockOperator*> Operators_;
    Array<BlockSmoother*> Smoothers_;

    mutable int current_level;

    mutable Array<Vector*> correction;
    mutable Array<Vector*> residual;

    mutable Vector res_aux;
    mutable Vector cor_cor;
    mutable Vector cor_aux;

    CGSolver *CoarseSolver;
    Solver *CoarsePrec_;
};

void MonolithicMultigrid::Mult(const Vector & x, Vector & y) const
{
    *residual.Last() = x;
    correction.Last()->SetDataAndSize(y.GetData(), y.Size());
    MG_Cycle();
}

void MonolithicMultigrid::MG_Cycle() const
{
    // PreSmoothing
    const BlockOperator& Operator_l = *Operators_[current_level];
    const BlockSmoother& Smoother_l = *Smoothers_[current_level];

    Vector& residual_l = *residual[current_level];
    Vector& correction_l = *correction[current_level];
    Vector help(residual_l.Size());
    help = 0.0;

    Smoother_l.Mult(residual_l, correction_l);

    Operator_l.Mult(correction_l, help);
    residual_l -= help;

    // Coarse grid correction
    if (current_level > 0)
    {
        const BlockOperator& P_l = *P_[current_level-1];

        P_l.MultTranspose(residual_l, *residual[current_level-1]);

        current_level--;
        MG_Cycle();
        current_level++;

        cor_cor.SetSize(residual_l.Size());
        P_l.Mult(*correction[current_level-1], cor_cor);
        correction_l += cor_cor;
        Operator_l.Mult(cor_cor, help);
        residual_l -= help;
    }
    else
    {
        cor_cor.SetSize(residual_l.Size());
        if (CoarseSolver)
        {
            CoarseSolver->Mult(residual_l, cor_cor);
            correction_l += cor_cor;
            Operator_l.Mult(cor_cor, help);
            residual_l -= help;
        }
    }

    // PostSmoothing
    Smoother_l.MultTranspose(residual_l, cor_cor);
    correction_l += cor_cor;
}

class Multigrid : public Solver
{
public:
    Multigrid(HypreParMatrix &Operator,
              const Array<HypreParMatrix*> &P,
              Solver *CoarsePrec=NULL)
        :
          Solver(Operator.GetNumRows()),
          P_(P),
          Operators_(P.Size()+1),
          Smoothers_(Operators_.Size()),
          current_level(Operators_.Size()-1),
          correction(Operators_.Size()),
          residual(Operators_.Size()),
          CoarseSolver(NULL),
          CoarsePrec_(CoarsePrec)
    {
        Operators_.Last() = &Operator;
        for (int l = Operators_.Size()-1; l > 0; l--)
        {
            // Two steps RAP
            unique_ptr<HypreParMatrix> PT( P[l-1]->Transpose() );
            unique_ptr<HypreParMatrix> AP( ParMult(Operators_[l], P[l-1]) );
            Operators_[l-1] = ParMult(PT.get(), AP.get());
            Operators_[l-1]->CopyRowStarts();
        }

        for (int l = 0; l < Operators_.Size(); l++)
        {
            Smoothers_[l] = new HypreSmoother(*Operators_[l]);
            correction[l] = new Vector(Operators_[l]->GetNumRows());
            residual[l] = new Vector(Operators_[l]->GetNumRows());
        }

        if (CoarsePrec)
        {
            CoarseSolver = new CGSolver(Operators_[0]->GetComm());
            CoarseSolver->SetRelTol(1e-8);
            CoarseSolver->SetMaxIter(50);
            CoarseSolver->SetPrintLevel(0);
            CoarseSolver->SetOperator(*Operators_[0]);
            CoarseSolver->SetPreconditioner(*CoarsePrec);
        }
    }

    virtual void Mult(const Vector & x, Vector & y) const;

    virtual void SetOperator(const Operator &op) { }

    ~Multigrid()
    {
        for (int l = 0; l < Operators_.Size(); l++)
        {
            delete Smoothers_[l];
            delete correction[l];
            delete residual[l];
        }
    }

private:
    void MG_Cycle() const;

    const Array<HypreParMatrix*> &P_;

    Array<HypreParMatrix*> Operators_;
    Array<HypreSmoother*> Smoothers_;

    mutable int current_level;

    mutable Array<Vector*> correction;
    mutable Array<Vector*> residual;

    mutable Vector res_aux;
    mutable Vector cor_cor;
    mutable Vector cor_aux;

    CGSolver *CoarseSolver;
    Solver *CoarsePrec_;
};

void Multigrid::Mult(const Vector & x, Vector & y) const
{
    *residual.Last() = x;
    correction.Last()->SetDataAndSize(y.GetData(), y.Size());
    MG_Cycle();
}

void Multigrid::MG_Cycle() const
{
    // PreSmoothing
    const HypreParMatrix& Operator_l = *Operators_[current_level];
    const HypreSmoother& Smoother_l = *Smoothers_[current_level];

    Vector& residual_l = *residual[current_level];
    Vector& correction_l = *correction[current_level];

    Smoother_l.Mult(residual_l, correction_l);
    Operator_l.Mult(-1.0, correction_l, 1.0, residual_l);

    // Coarse grid correction
    if (current_level > 0)
    {
        const HypreParMatrix& P_l = *P_[current_level-1];

        P_l.MultTranspose(residual_l, *residual[current_level-1]);

        current_level--;
        MG_Cycle();
        current_level++;

        cor_cor.SetSize(residual_l.Size());
        P_l.Mult(*correction[current_level-1], cor_cor);
        correction_l += cor_cor;
        Operator_l.Mult(-1.0, cor_cor, 1.0, residual_l);
    }
    else
    {
        cor_cor.SetSize(residual_l.Size());
        if (CoarseSolver)
        {
            CoarseSolver->Mult(residual_l, cor_cor);
            correction_l += cor_cor;
            Operator_l.Mult(-1.0, cor_cor, 1.0, residual_l);
        }
    }

    // PostSmoothing
    Smoother_l.Mult(residual_l, cor_cor);
    correction_l += cor_cor;
}

SparseMatrix * RemoveZeroEntries(const SparseMatrix& in)
{
    int * I = in.GetI();
    int * J = in.GetJ();
    double * Data = in.GetData();
    double * End = Data+in.NumNonZeroElems();

    int nnz = 0;
    for (double * data_ptr = Data; data_ptr != End; data_ptr++)
    {
        if (*data_ptr != 0)
            nnz++;
    }

    int * outI = new int[in.Height()+1];
    int * outJ = new int[nnz];
    double * outData = new double[nnz];
    nnz = 0;
    for (int i = 0; i < in.Height(); i++)
    {
        outI[i] = nnz;
        for (int j = I[i]; j < I[i+1]; j++)
        {
            if (Data[j] !=0)
            {
                outJ[nnz] = J[j];
                outData[nnz++] = Data[j];
            }
        }
    }
    outI[in.Height()] = nnz;

    return new SparseMatrix(outI, outJ, outData, in.Height(), in.Width());
}
