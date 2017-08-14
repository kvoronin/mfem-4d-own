// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Implementation of GridFunction

#include "gridfunc.hpp"
#include "../mesh/nurbs.hpp"

#include <limits>
#include <cstring>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace mfem
{

using namespace std;

GridFunction::GridFunction(Mesh *m, std::istream &input)
   : Vector()
{
   const int bufflen = 256;
   char buff[bufflen];
   int vdim;

   input >> std::ws;
   input.getline(buff, bufflen);  // 'FiniteElementSpace'
   if (strcmp(buff, "FiniteElementSpace"))
   {
      mfem_error("GridFunction::GridFunction():"
                 " input stream is not a GridFunction!");
   }
   input.getline(buff, bufflen, ' '); // 'FiniteElementCollection:'
   input >> std::ws;
   input.getline(buff, bufflen);
   fec = FiniteElementCollection::New(buff);
   input.getline(buff, bufflen, ' '); // 'VDim:'
   input >> vdim;
   input.getline(buff, bufflen, ' '); // 'Ordering:'
   int ordering;
   input >> ordering;
   input.getline(buff, bufflen); // read the empty line
   fes = new FiniteElementSpace(m, fec, vdim, ordering);
   Vector::Load(input, fes->GetVSize());
   sequence = 0;
}

GridFunction::GridFunction(Mesh *m, GridFunction *gf_array[], int num_pieces)
{
   // all GridFunctions must have the same FE collection, vdim, ordering
   int vdim, ordering;

   fes = gf_array[0]->FESpace();
   fec = FiniteElementCollection::New(fes->FEColl()->Name());
   vdim = fes->GetVDim();
   ordering = fes->GetOrdering();
   fes = new FiniteElementSpace(m, fec, vdim, ordering);
   SetSize(fes->GetVSize());

   if (m->NURBSext)
   {
      m->NURBSext->MergeGridFunctions(gf_array, num_pieces, *this);
      return;
   }

   int g_ndofs  = fes->GetNDofs();
   int g_nvdofs = fes->GetNVDofs();
   int g_nedofs = fes->GetNEDofs();
   int g_nfdofs = fes->GetNFDofs();
   int g_nddofs = g_ndofs - (g_nvdofs + g_nedofs + g_nfdofs);
   int vi, ei, fi, di;
   vi = ei = fi = di = 0;
   for (int i = 0; i < num_pieces; i++)
   {
      FiniteElementSpace *l_fes = gf_array[i]->FESpace();
      int l_ndofs  = l_fes->GetNDofs();
      int l_nvdofs = l_fes->GetNVDofs();
      int l_nedofs = l_fes->GetNEDofs();
      int l_nfdofs = l_fes->GetNFDofs();
      int l_nddofs = l_ndofs - (l_nvdofs + l_nedofs + l_nfdofs);
      const double *l_data = gf_array[i]->GetData();
      double *g_data = data;
      if (ordering == Ordering::byNODES)
      {
         for (int d = 0; d < vdim; d++)
         {
            memcpy(g_data+vi, l_data, l_nvdofs*sizeof(double));
            l_data += l_nvdofs;
            g_data += g_nvdofs;
            memcpy(g_data+ei, l_data, l_nedofs*sizeof(double));
            l_data += l_nedofs;
            g_data += g_nedofs;
            memcpy(g_data+fi, l_data, l_nfdofs*sizeof(double));
            l_data += l_nfdofs;
            g_data += g_nfdofs;
            memcpy(g_data+di, l_data, l_nddofs*sizeof(double));
            l_data += l_nddofs;
            g_data += g_nddofs;
         }
      }
      else
      {
         memcpy(g_data+vdim*vi, l_data, vdim*l_nvdofs*sizeof(double));
         l_data += vdim*l_nvdofs;
         g_data += vdim*g_nvdofs;
         memcpy(g_data+vdim*ei, l_data, vdim*l_nedofs*sizeof(double));
         l_data += vdim*l_nedofs;
         g_data += vdim*g_nedofs;
         memcpy(g_data+vdim*fi, l_data, vdim*l_nfdofs*sizeof(double));
         l_data += vdim*l_nfdofs;
         g_data += vdim*g_nfdofs;
         memcpy(g_data+vdim*di, l_data, vdim*l_nddofs*sizeof(double));
         l_data += vdim*l_nddofs;
         g_data += vdim*g_nddofs;
      }
      vi += l_nvdofs;
      ei += l_nedofs;
      fi += l_nfdofs;
      di += l_nddofs;
   }
   sequence = 0;
}

void GridFunction::Destroy()
{
   if (fec)
   {
      delete fes;
      delete fec;
      fec = NULL;
   }
}

void GridFunction::Update()
{
   const Operator *T = fes->GetUpdateOperator();

   if (fes->GetSequence() == sequence)
   {
      return; // space and grid function are in sync, no-op
   }
   if (fes->GetSequence() != sequence + 1)
   {
      MFEM_ABORT("Error in update sequence. GridFunction needs to be updated "
                 "right after the space is updated.");
   }
   sequence = fes->GetSequence();

   if (T)
   {
      Vector tmp(T->Height());
      T->Mult(*this, tmp);
      *this = tmp;
   }
   else
   {
      SetSize(fes->GetVSize());
   }
}

void GridFunction::SetSpace(FiniteElementSpace *f)
{
   Destroy();
   fes = f;
   SetSize(fes->GetVSize());
   sequence = fes->GetSequence();
}

void GridFunction::MakeRef(FiniteElementSpace *f, double *v)
{
   Destroy();
   fes = f;
   NewDataAndSize(v, fes->GetVSize());
   sequence = fes->GetSequence();
}

void GridFunction::MakeRef(FiniteElementSpace *f, Vector &v, int v_offset)
{
   MFEM_ASSERT(v.Size() >= v_offset + f->GetVSize(), "");
   Destroy();
   fes = f;
   NewDataAndSize((double *)v + v_offset, fes->GetVSize());
   sequence = fes->GetSequence();
}


void GridFunction::SumFluxAndCount(BilinearFormIntegrator &blfi,
                                   GridFunction &flux,
                                   Array<int>& count,
                                   int wcoef,
                                   int subdomain)
{
   GridFunction &u = *this;

   ElementTransformation *Transf;

   FiniteElementSpace *ufes = u.FESpace();
   FiniteElementSpace *ffes = flux.FESpace();

   int nfe = ufes->GetNE();
   Array<int> udofs;
   Array<int> fdofs;
   Vector ul, fl;

   flux = 0.0;
   count = 0;

   for (int i = 0; i < nfe; i++)
   {
      if (subdomain >= 0 && ufes->GetAttribute(i) != subdomain)
      {
         continue;
      }

      ufes->GetElementVDofs(i, udofs);
      ffes->GetElementVDofs(i, fdofs);

      u.GetSubVector(udofs, ul);

      Transf = ufes->GetElementTransformation(i);
      blfi.ComputeElementFlux(*ufes->GetFE(i), *Transf, ul,
                              *ffes->GetFE(i), fl, wcoef);

      flux.AddElementVector(fdofs, fl);

      FiniteElementSpace::AdjustVDofs(fdofs);
      for (int j = 0; j < fdofs.Size(); j++)
      {
         count[fdofs[j]]++;
      }
   }
}

void GridFunction::ComputeFlux(BilinearFormIntegrator &blfi,
                               GridFunction &flux, int wcoef,
                               int subdomain)
{
   Array<int> count(flux.Size());

   SumFluxAndCount(blfi, flux, count, wcoef, subdomain);

   // complete averaging
   for (int i = 0; i < count.Size(); i++)
   {
      if (count[i] != 0) { flux(i) /= count[i]; }
   }
}

int GridFunction::VectorDim() const
{
   const FiniteElement *fe;
   if (!fes->GetNE())
   {
      const FiniteElementCollection *fec = fes->FEColl();
      static const int geoms[3] =
      { Geometry::SEGMENT, Geometry::TRIANGLE, Geometry::TETRAHEDRON };
      fe = fec->FiniteElementForGeometry(geoms[fes->GetMesh()->Dimension()-1]);
   }
   else
   {
      fe = fes->GetFE(0);
   }
   if (fe->GetRangeType() == FiniteElement::SCALAR)
   {
      return fes->GetVDim();
   }
   return fes->GetVDim()*fes->GetMesh()->SpaceDimension();
}

void GridFunction::GetTrueDofs(Vector &tv) const
{
   const SparseMatrix *R = fes->GetRestrictionMatrix();
   if (!R)
   {
      // R is identity -> make tv a reference to *this
      tv.NewDataAndSize(data, size);
   }
   else
   {
      tv.SetSize(R->Height());
      R->Mult(*this, tv);
   }
}

void GridFunction::SetFromTrueDofs(const Vector &tv)
{
   MFEM_ASSERT(tv.Size() == fes->GetTrueVSize(), "invalid input");
   const SparseMatrix *cP = fes->GetConformingProlongation();
   if (!cP)
   {
      if (tv.GetData() != data)
      {
         *this = tv;
      }
   }
   else
   {
      cP->Mult(tv, *this);
   }
}

void GridFunction::GetNodalValues(int i, Array<double> &nval, int vdim) const
{
   Array<int> vdofs;

   int k;

   fes->GetElementVDofs(i, vdofs);
   const FiniteElement *FElem = fes->GetFE(i);
   const IntegrationRule *ElemVert =
      Geometries.GetVertices(FElem->GetGeomType());
   int dof = FElem->GetDof();
   int n = ElemVert->GetNPoints();
   nval.SetSize(n);
   vdim--;
   Vector loc_data;
   GetSubVector(vdofs, loc_data);

   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
      MFEM_ASSERT(FElem->GetMapType() == FiniteElement::VALUE,
                  "invalid FE map type");
      Vector shape(dof);
      for (k = 0; k < n; k++)
      {
         FElem->CalcShape(ElemVert->IntPoint(k), shape);
         nval[k] = shape * ((const double *)loc_data + dof * vdim);
      }
   }
   else
   {
      ElementTransformation *Tr = fes->GetElementTransformation(i);
      DenseMatrix vshape(dof, FElem->GetDim());
      for (k = 0; k < n; k++)
      {
         Tr->SetIntPoint(&ElemVert->IntPoint(k));
         FElem->CalcVShape(*Tr, vshape);
         nval[k] = loc_data * (&vshape(0,vdim));
      }
   }
}

double GridFunction::GetValue(int i, const IntegrationPoint &ip, int vdim)
const
{
   Array<int> dofs;
   fes->GetElementDofs(i, dofs);
   fes->DofsToVDofs(vdim-1, dofs);
   Vector DofVal(dofs.Size()), LocVec;
   const FiniteElement *fe = fes->GetFE(i);
   MFEM_ASSERT(fe->GetMapType() == FiniteElement::VALUE, "invalid FE map type");
   fe->CalcShape(ip, DofVal);
   GetSubVector(dofs, LocVec);

   return (DofVal * LocVec);
}

void GridFunction::GetVectorValue(int i, const IntegrationPoint &ip,
                                  Vector &val) const
{
   const FiniteElement *FElem = fes->GetFE(i);
   int dof = FElem->GetDof();
   Array<int> vdofs;
   fes->GetElementVDofs(i, vdofs);
   Vector loc_data;
   GetSubVector(vdofs, loc_data);
   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
      MFEM_ASSERT(FElem->GetMapType() == FiniteElement::VALUE,
                  "invalid FE map type");
      Vector shape(dof);
      FElem->CalcShape(ip, shape);
      int vdim = fes->GetVDim();
      val.SetSize(vdim);
      for (int k = 0; k < vdim; k++)
      {
         val(k) = shape * ((const double *)loc_data + dof * k);
      }
   }
   else
   {
      int spaceDim = fes->GetMesh()->SpaceDimension();
      DenseMatrix vshape(dof, spaceDim);
      ElementTransformation *Tr = fes->GetElementTransformation(i);
      Tr->SetIntPoint(&ip);
      FElem->CalcVShape(*Tr, vshape);
      val.SetSize(spaceDim);
      vshape.MultTranspose(loc_data, val);
   }
}

void GridFunction::GetValues(int i, const IntegrationRule &ir, Vector &vals,
                             int vdim)
const
{
   Array<int> dofs;
   int n = ir.GetNPoints();
   vals.SetSize(n);
   fes->GetElementDofs(i, dofs);
   fes->DofsToVDofs(vdim-1, dofs);
   const FiniteElement *FElem = fes->GetFE(i);
   MFEM_ASSERT(FElem->GetMapType() == FiniteElement::VALUE,
               "invalid FE map type");
   int dof = FElem->GetDof();
   Vector DofVal(dof), loc_data(dof);
   GetSubVector(dofs, loc_data);
   for (int k = 0; k < n; k++)
   {
      FElem->CalcShape(ir.IntPoint(k), DofVal);
      vals(k) = DofVal * loc_data;
   }
}

void GridFunction::GetValues(int i, const IntegrationRule &ir, Vector &vals,
                             DenseMatrix &tr, int vdim)
const
{
   ElementTransformation *ET;
   ET = fes->GetElementTransformation(i);
   ET->Transform(ir, tr);

   GetValues(i, ir, vals, vdim);
}

int GridFunction::GetFaceValues(int i, int side, const IntegrationRule &ir,
                                Vector &vals, DenseMatrix &tr,
                                int vdim) const
{
   int n, dir;
   FaceElementTransformations *Transf;

   n = ir.GetNPoints();
   IntegrationRule eir(n);  // ---
   if (side == 2) // automatic choice of side
   {
      Transf = fes->GetMesh()->GetFaceElementTransformations(i, 0);
      if (Transf->Elem2No < 0 ||
          fes->GetAttribute(Transf->Elem1No) <=
          fes->GetAttribute(Transf->Elem2No))
      {
         dir = 0;
      }
      else
      {
         dir = 1;
      }
   }
   else
   {
      if (side == 1 && !fes->GetMesh()->FaceIsInterior(i))
      {
         dir = 0;
      }
      else
      {
         dir = side;
      }
   }
   if (dir == 0)
   {
      Transf = fes->GetMesh()->GetFaceElementTransformations(i, 4);
      Transf->Loc1.Transform(ir, eir);
      GetValues(Transf->Elem1No, eir, vals, tr, vdim);
   }
   else
   {
      Transf = fes->GetMesh()->GetFaceElementTransformations(i, 8);
      Transf->Loc2.Transform(ir, eir);
      GetValues(Transf->Elem2No, eir, vals, tr, vdim);
   }

   return dir;
}

void GridFunction::GetVectorValues(ElementTransformation &T,
                                   const IntegrationRule &ir,
                                   DenseMatrix &vals) const
{
   const FiniteElement *FElem = fes->GetFE(T.ElementNo);
   int dof = FElem->GetDof();
   Array<int> vdofs;
   fes->GetElementVDofs(T.ElementNo, vdofs);
   Vector loc_data;
   GetSubVector(vdofs, loc_data);
   int nip = ir.GetNPoints();
   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
      MFEM_ASSERT(FElem->GetMapType() == FiniteElement::VALUE,
                  "invalid FE map type");
      Vector shape(dof);
      int vdim = fes->GetVDim();
      vals.SetSize(vdim, nip);
      for (int j = 0; j < nip; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         FElem->CalcShape(ip, shape);
         for (int k = 0; k < vdim; k++)
         {
            vals(k,j) = shape * ((const double *)loc_data + dof * k);
         }
      }
   }
   else
   {
      int spaceDim = fes->GetMesh()->SpaceDimension();
      DenseMatrix vshape;
      vshape.SetSize(dof, spaceDim);
      vals.SetSize(spaceDim, nip);
      if (spaceDim == 4 && FElem->GetMapType() == FiniteElement::H_DIV_SKEW)
      {
          vshape.SetSize(dof, spaceDim*spaceDim);
          vals.SetSize(spaceDim*spaceDim, nip);
      }
      Vector val_j;
      for (int j = 0; j < nip; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         T.SetIntPoint(&ip);
         FElem->CalcVShape(T, vshape);
         vals.GetColumnReference(j, val_j);
         vshape.MultTranspose(loc_data, val_j);
      }
   }
}

void GridFunction::GetVectorValues(int i, const IntegrationRule &ir,
                                   DenseMatrix &vals, DenseMatrix &tr) const
{
   ElementTransformation *Tr = fes->GetElementTransformation(i);
   Tr->Transform(ir, tr);

   GetVectorValues(*Tr, ir, vals);
}

int GridFunction::GetFaceVectorValues(
   int i, int side, const IntegrationRule &ir,
   DenseMatrix &vals, DenseMatrix &tr) const
{
   int n, di;
   FaceElementTransformations *Transf;

   n = ir.GetNPoints();
   IntegrationRule eir(n);  // ---
   Transf = fes->GetMesh()->GetFaceElementTransformations(i, 0);
   if (side == 2)
   {
      if (Transf->Elem2No < 0 ||
          fes->GetAttribute(Transf->Elem1No) <=
          fes->GetAttribute(Transf->Elem2No))
      {
         di = 0;
      }
      else
      {
         di = 1;
      }
   }
   else
   {
      di = side;
   }
   if (di == 0)
   {
      Transf = fes->GetMesh()->GetFaceElementTransformations(i, 4);
      Transf->Loc1.Transform(ir, eir);
      GetVectorValues(Transf->Elem1No, eir, vals, tr);
   }
   else
   {
      Transf = fes->GetMesh()->GetFaceElementTransformations(i, 8);
      Transf->Loc2.Transform(ir, eir);
      GetVectorValues(Transf->Elem2No, eir, vals, tr);
   }

   return di;
}

void GridFunction::GetValuesFrom(GridFunction &orig_func)
{
   // Without averaging ...

   FiniteElementSpace *orig_fes = orig_func.FESpace();
   Array<int> vdofs, orig_vdofs;
   Vector shape, loc_values, orig_loc_values;
   int i, j, d, ne, dof, odof, vdim;

   ne = fes->GetNE();
   vdim = fes->GetVDim();
   for (i = 0; i < ne; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      orig_fes->GetElementVDofs(i, orig_vdofs);
      orig_func.GetSubVector(orig_vdofs, orig_loc_values);
      const FiniteElement *fe = fes->GetFE(i);
      const FiniteElement *orig_fe = orig_fes->GetFE(i);
      dof = fe->GetDof();
      odof = orig_fe->GetDof();
      loc_values.SetSize(dof * vdim);
      shape.SetSize(odof);
      const IntegrationRule &ir = fe->GetNodes();
      for (j = 0; j < dof; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         orig_fe->CalcShape(ip, shape);
         for (d = 0; d < vdim; d++)
         {
            loc_values(d*dof+j) =
               shape * ((const double *)orig_loc_values + d * odof) ;
         }
      }
      SetSubVector(vdofs, loc_values);
   }
}

void GridFunction::GetBdrValuesFrom(GridFunction &orig_func)
{
   // Without averaging ...

   FiniteElementSpace *orig_fes = orig_func.FESpace();
   Array<int> vdofs, orig_vdofs;
   Vector shape, loc_values, orig_loc_values;
   int i, j, d, nbe, dof, odof, vdim;

   nbe = fes->GetNBE();
   vdim = fes->GetVDim();
   for (i = 0; i < nbe; i++)
   {
      fes->GetBdrElementVDofs(i, vdofs);
      orig_fes->GetBdrElementVDofs(i, orig_vdofs);
      orig_func.GetSubVector(orig_vdofs, orig_loc_values);
      const FiniteElement *fe = fes->GetBE(i);
      const FiniteElement *orig_fe = orig_fes->GetBE(i);
      dof = fe->GetDof();
      odof = orig_fe->GetDof();
      loc_values.SetSize(dof * vdim);
      shape.SetSize(odof);
      const IntegrationRule &ir = fe->GetNodes();
      for (j = 0; j < dof; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         orig_fe->CalcShape(ip, shape);
         for (d = 0; d < vdim; d++)
         {
            loc_values(d*dof+j) =
               shape * ((const double *)orig_loc_values + d * odof);
         }
      }
      SetSubVector(vdofs, loc_values);
   }
}

void GridFunction::GetVectorFieldValues(
   int i, const IntegrationRule &ir, DenseMatrix &vals,
   DenseMatrix &tr, int comp) const
{
   Array<int> vdofs;
   ElementTransformation *transf;

   int d, j, k, n, sdim, dof, ind;

   n = ir.GetNPoints();
   fes->GetElementVDofs(i, vdofs);
   const FiniteElement *fe = fes->GetFE(i);
   dof = fe->GetDof();
   sdim = fes->GetMesh()->SpaceDimension();
   int *dofs = &vdofs[comp*dof];
   transf = fes->GetElementTransformation(i);
   transf->Transform(ir, tr);
   vals.SetSize(n, sdim);
   DenseMatrix vshape(dof, sdim);
   double a;
   for (k = 0; k < n; k++)
   {
      const IntegrationPoint &ip = ir.IntPoint(k);
      transf->SetIntPoint(&ip);
      fe->CalcVShape(*transf, vshape);
      for (d = 0; d < sdim; d++)
      {
         a = 0.0;
         for (j = 0; j < dof; j++)
            if ( (ind=dofs[j]) >= 0 )
            {
               a += vshape(j, d) * data[ind];
            }
            else
            {
               a -= vshape(j, d) * data[-1-ind];
            }
         vals(k, d) = a;
      }
   }
}

void GridFunction::ReorderByNodes()
{
   if (fes->GetOrdering() == Ordering::byNODES)
   {
      return;
   }

   int i, j, k;
   int vdim = fes->GetVDim();
   int ndofs = fes->GetNDofs();
   double *temp = new double[size];

   k = 0;
   for (j = 0; j < ndofs; j++)
      for (i = 0; i < vdim; i++)
      {
         temp[j+i*ndofs] = data[k++];
      }

   for (i = 0; i < size; i++)
   {
      data[i] = temp[i];
   }

   delete [] temp;
}

void GridFunction::GetVectorFieldNodalValues(Vector &val, int comp) const
{
   int i, k;
   Array<int> overlap(fes->GetNV());
   Array<int> vertices;
   DenseMatrix vals, tr;

   val.SetSize(overlap.Size());
   overlap = 0;
   val = 0.0;

   comp--;
   for (i = 0; i < fes->GetNE(); i++)
   {
      const IntegrationRule *ir =
         Geometries.GetVertices(fes->GetFE(i)->GetGeomType());
      fes->GetElementVertices(i, vertices);
      GetVectorFieldValues(i, *ir, vals, tr);
      for (k = 0; k < ir->GetNPoints(); k++)
      {
         val(vertices[k]) += vals(k, comp);
         overlap[vertices[k]]++;
      }
   }

   for (i = 0; i < overlap.Size(); i++)
   {
      val(i) /= overlap[i];
   }
}

void GridFunction::ProjectVectorFieldOn(GridFunction &vec_field, int comp)
{
   FiniteElementSpace *new_fes = vec_field.FESpace();

   int d, i, k, ind, dof, sdim;
   Array<int> overlap(new_fes->GetVSize());
   Array<int> new_vdofs;
   DenseMatrix vals, tr;

   sdim = fes->GetMesh()->SpaceDimension();
   overlap = 0;
   vec_field = 0.0;

   for (i = 0; i < new_fes->GetNE(); i++)
   {
      const FiniteElement *fe = new_fes->GetFE(i);
      const IntegrationRule &ir = fe->GetNodes();
      GetVectorFieldValues(i, ir, vals, tr, comp);
      new_fes->GetElementVDofs(i, new_vdofs);
      dof = fe->GetDof();
      for (d = 0; d < sdim; d++)
      {
         for (k = 0; k < dof; k++)
         {
            if ( (ind=new_vdofs[dof*d+k]) < 0 )
            {
               ind = -1-ind, vals(k, d) = - vals(k, d);
            }
            vec_field(ind) += vals(k, d);
            overlap[ind]++;
         }
      }
   }

   for (i = 0; i < overlap.Size(); i++)
   {
      vec_field(i) /= overlap[i];
   }
}

void GridFunction::GetDerivative(int comp, int der_comp, GridFunction &der)
{
   FiniteElementSpace * der_fes = der.FESpace();
   ElementTransformation * transf;
   Array<int> overlap(der_fes->GetVSize());
   Array<int> der_dofs, vdofs;
   DenseMatrix dshape, inv_jac;
   Vector pt_grad, loc_func;
   int i, j, k, dim, dof, der_dof, ind;
   double a;

   for (i = 0; i < overlap.Size(); i++)
   {
      overlap[i] = 0;
   }
   der = 0.0;

   comp--;
   for (i = 0; i < der_fes->GetNE(); i++)
   {
      const FiniteElement *der_fe = der_fes->GetFE(i);
      const FiniteElement *fe = fes->GetFE(i);
      const IntegrationRule &ir = der_fe->GetNodes();
      der_fes->GetElementDofs(i, der_dofs);
      fes->GetElementVDofs(i, vdofs);
      dim = fe->GetDim();
      dof = fe->GetDof();
      der_dof = der_fe->GetDof();
      dshape.SetSize(dof, dim);
      inv_jac.SetSize(dim);
      pt_grad.SetSize(dim);
      loc_func.SetSize(dof);
      transf = fes->GetElementTransformation(i);
      for (j = 0; j < dof; j++)
         loc_func(j) = ( (ind=vdofs[comp*dof+j]) >= 0 ) ?
                       (data[ind]) : (-data[-1-ind]);
      for (k = 0; k < der_dof; k++)
      {
         const IntegrationPoint &ip = ir.IntPoint(k);
         fe->CalcDShape(ip, dshape);
         dshape.MultTranspose(loc_func, pt_grad);
         transf->SetIntPoint(&ip);
         CalcInverse(transf->Jacobian(), inv_jac);
         a = 0.0;
         for (j = 0; j < dim; j++)
         {
            a += inv_jac(j, der_comp) * pt_grad(j);
         }
         der(der_dofs[k]) += a;
         overlap[der_dofs[k]]++;
      }
   }

   for (i = 0; i < overlap.Size(); i++)
   {
      der(i) /= overlap[i];
   }
}


void GridFunction::GetVectorGradientHat(
   ElementTransformation &T, DenseMatrix &gh)
{
   int elNo = T.ElementNo;
   const FiniteElement *FElem = fes->GetFE(elNo);
   int dim = FElem->GetDim(), dof = FElem->GetDof();
   Array<int> vdofs;
   fes->GetElementVDofs(elNo, vdofs);
   Vector loc_data;
   GetSubVector(vdofs, loc_data);
   // assuming scalar FE
   int vdim = fes->GetVDim();
   DenseMatrix dshape(dof, dim);
   FElem->CalcDShape(T.GetIntPoint(), dshape);
   gh.SetSize(vdim, dim);
   DenseMatrix loc_data_mat(loc_data.GetData(), dof, vdim);
   MultAtB(loc_data_mat, dshape, gh);
}

double GridFunction::GetDivergence(ElementTransformation &tr)
{
   double div_v;
   int elNo = tr.ElementNo;
   const FiniteElement *FElem = fes->GetFE(elNo);
   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
      MFEM_ASSERT(FElem->GetMapType() == FiniteElement::VALUE,
                  "invalid FE map type");
      DenseMatrix grad_hat;
      GetVectorGradientHat(tr, grad_hat);
      const DenseMatrix &J = tr.Jacobian();
      DenseMatrix Jinv(J.Width(), J.Height());
      CalcInverse(J, Jinv);
      div_v = 0.0;
      for (int i = 0; i < Jinv.Width(); i++)
      {
         for (int j = 0; j < Jinv.Height(); j++)
         {
            div_v += grad_hat(i, j) * Jinv(j, i);
         }
      }
   }
   else
   {
      // Assuming RT-type space
      Array<int> dofs;
      fes->GetElementDofs(elNo, dofs);
      Vector loc_data, divshape(FElem->GetDof());
      GetSubVector(dofs, loc_data);
      FElem->CalcDivShape(tr.GetIntPoint(), divshape);
      div_v = (loc_data * divshape) / tr.Weight();
   }
   return div_v;
}

void GridFunction::GetCurl(ElementTransformation &tr, Vector &curl)
{
   int elNo = tr.ElementNo;
   const FiniteElement *FElem = fes->GetFE(elNo);
   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
      MFEM_ASSERT(FElem->GetMapType() == FiniteElement::VALUE,
                  "invalid FE map type");
      DenseMatrix grad_hat;
      GetVectorGradientHat(tr, grad_hat);
      const DenseMatrix &J = tr.Jacobian();
      DenseMatrix Jinv(J.Width(), J.Height());
      CalcInverse(J, Jinv);
      DenseMatrix grad(grad_hat.Height(), Jinv.Width()); // vdim x FElem->Dim
      Mult(grad_hat, Jinv, grad);
      MFEM_ASSERT(grad.Height() == grad.Width(), "");
      if (grad.Height() == 3)
      {
         curl.SetSize(3);
         curl(0) = grad(2,1) - grad(1,2);
         curl(1) = grad(0,2) - grad(2,0);
         curl(2) = grad(1,0) - grad(0,1);
      }
      else if (grad.Height() == 2)
      {
         curl.SetSize(1);
         curl(0) = grad(1,0) - grad(0,1);
      }
   }
   else
   {
      // Assuming ND-type space
      Array<int> dofs;
      fes->GetElementDofs(elNo, dofs);
      Vector loc_data;
      GetSubVector(dofs, loc_data);
      DenseMatrix curl_shape(FElem->GetDof(), FElem->GetDim() == 3 ? 3 : 1);
      FElem->CalcCurlShape(tr.GetIntPoint(), curl_shape);
      curl.SetSize(curl_shape.Width());
      if (curl_shape.Width() == 3)
      {
         double curl_hat[3];
         curl_shape.MultTranspose(loc_data, curl_hat);
         tr.Jacobian().Mult(curl_hat, curl);
      }
      else
      {
         curl_shape.MultTranspose(loc_data, curl);
      }
      curl /= tr.Weight();
   }
}

void GridFunction::GetGradient(ElementTransformation &tr, Vector &grad)
{
   int elNo = tr.ElementNo;
   const FiniteElement *fe = fes->GetFE(elNo);
   MFEM_ASSERT(fe->GetMapType() == FiniteElement::VALUE, "invalid FE map type");
   int dim = fe->GetDim(), dof = fe->GetDof();
   DenseMatrix dshape(dof, dim), Jinv(dim);
   Vector lval, gh(dim);
   Array<int> dofs;

   grad.SetSize(dim);
   fes->GetElementDofs(elNo, dofs);
   GetSubVector(dofs, lval);
   fe->CalcDShape(tr.GetIntPoint(), dshape);
   dshape.MultTranspose(lval, gh);
   CalcInverse(tr.Jacobian(), Jinv);
   Jinv.MultTranspose(gh, grad);
}

void GridFunction::GetGradients(const int elem, const IntegrationRule &ir,
                                DenseMatrix &grad)
{
   const FiniteElement *fe = fes->GetFE(elem);
   MFEM_ASSERT(fe->GetMapType() == FiniteElement::VALUE, "invalid FE map type");
   ElementTransformation *Tr = fes->GetElementTransformation(elem);
   DenseMatrix dshape(fe->GetDof(), fe->GetDim());
   DenseMatrix Jinv(fe->GetDim());
   Vector lval, gh(fe->GetDim()), gcol;
   Array<int> dofs;
   fes->GetElementDofs(elem, dofs);
   GetSubVector(dofs, lval);
   grad.SetSize(fe->GetDim(), ir.GetNPoints());
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      fe->CalcDShape(ip, dshape);
      dshape.MultTranspose(lval, gh);
      Tr->SetIntPoint(&ip);
      grad.GetColumnReference(i, gcol);
      CalcInverse(Tr->Jacobian(), Jinv);
      Jinv.MultTranspose(gh, gcol);
   }
}

void GridFunction::GetVectorGradient(
   ElementTransformation &tr, DenseMatrix &grad)
{
   MFEM_ASSERT(fes->GetFE(tr.ElementNo)->GetMapType() == FiniteElement::VALUE,
               "invalid FE map type");
   DenseMatrix grad_hat;
   GetVectorGradientHat(tr, grad_hat);
   const DenseMatrix &J = tr.Jacobian();
   DenseMatrix Jinv(J.Width(), J.Height());
   CalcInverse(J, Jinv);
   grad.SetSize(grad_hat.Height(), Jinv.Width());
   Mult(grad_hat, Jinv, grad);
}

void GridFunction::GetElementAverages(GridFunction &avgs)
{
   MassIntegrator Mi;
   DenseMatrix loc_mass;
   Array<int> te_dofs, tr_dofs;
   Vector loc_avgs, loc_this;
   Vector int_psi(avgs.Size());

   avgs = 0.0;
   int_psi = 0.0;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      Mi.AssembleElementMatrix2(*fes->GetFE(i), *avgs.FESpace()->GetFE(i),
                                *fes->GetElementTransformation(i), loc_mass);
      fes->GetElementDofs(i, tr_dofs);
      avgs.FESpace()->GetElementDofs(i, te_dofs);
      GetSubVector(tr_dofs, loc_this);
      loc_avgs.SetSize(te_dofs.Size());
      loc_mass.Mult(loc_this, loc_avgs);
      avgs.AddElementVector(te_dofs, loc_avgs);
      loc_this = 1.0; // assume the local basis for 'this' sums to 1
      loc_mass.Mult(loc_this, loc_avgs);
      int_psi.AddElementVector(te_dofs, loc_avgs);
   }
   for (int i = 0; i < avgs.Size(); i++)
   {
      avgs(i) /= int_psi(i);
   }
}

void GridFunction::ProjectGridFunction(const GridFunction &src)
{
   // Assuming that the projection matrix is the same for all elements
   Mesh *mesh = fes->GetMesh();
   DenseMatrix P;

   if (!fes->GetNE())
   {
      return;
   }

   fes->GetFE(0)->Project(*src.fes->GetFE(0),
                          *mesh->GetElementTransformation(0), P);
   int vdim = fes->GetVDim();
   if (vdim != src.fes->GetVDim())
      mfem_error("GridFunction::ProjectGridFunction() :"
                 " incompatible vector dimensions!");
   Array<int> src_vdofs, dest_vdofs;
   Vector src_lvec, dest_lvec(vdim*P.Height());

   for (int i = 0; i < mesh->GetNE(); i++)
   {
      src.fes->GetElementVDofs(i, src_vdofs);
      src.GetSubVector(src_vdofs, src_lvec);
      for (int vd = 0; vd < vdim; vd++)
      {
         P.Mult(&src_lvec[vd*P.Width()], &dest_lvec[vd*P.Height()]);
      }
      fes->GetElementVDofs(i, dest_vdofs);
      SetSubVector(dest_vdofs, dest_lvec);
   }
}

void GridFunction::ImposeBounds(int i, const Vector &weights,
                                const Vector &_lo, const Vector &_hi)
{
   Array<int> vdofs;
   fes->GetElementVDofs(i, vdofs);
   int size = vdofs.Size();
   Vector vals, new_vals(size);
   GetSubVector(vdofs, vals);

   MFEM_ASSERT(weights.Size() == size, "Different # of weights and dofs.");
   MFEM_ASSERT(_lo.Size() == size, "Different # of lower bounds and dofs.");
   MFEM_ASSERT(_hi.Size() == size, "Different # of upper bounds and dofs.");

   int max_iter = 30;
   double tol = 1.e-12;
   SLBQPOptimizer slbqp;
   slbqp.SetMaxIter(max_iter);
   slbqp.SetAbsTol(1.0e-18);
   slbqp.SetRelTol(tol);
   slbqp.SetBounds(_lo, _hi);
   slbqp.SetLinearConstraint(weights, weights * vals);
   slbqp.SetPrintLevel(0); // print messages only if not converged
   slbqp.Mult(vals, new_vals);

   SetSubVector(vdofs, new_vals);
}

void GridFunction::ImposeBounds(int i, const Vector &weights,
                                double _min, double _max)
{
   Array<int> vdofs;
   fes->GetElementVDofs(i, vdofs);
   int size = vdofs.Size();
   Vector vals, new_vals(size);
   GetSubVector(vdofs, vals);

   double max_val = vals.Max();
   double min_val = vals.Min();

   if (max_val <= _min)
   {
      new_vals = _min;
      SetSubVector(vdofs, new_vals);
      return;
   }

   if (_min <= min_val && max_val <= _max)
   {
      return;
   }

   Vector minv(size), maxv(size);
   minv = (_min > min_val) ? _min : min_val;
   maxv = (_max < max_val) ? _max : max_val;

   ImposeBounds(i, weights, minv, maxv);
}

void GridFunction::GetNodalValues(Vector &nval, int vdim) const
{
   int i, j;
   Array<int> vertices;
   Array<double> values;
   Array<int> overlap(fes->GetNV());
   nval.SetSize(fes->GetNV());

   nval = 0.0;
   overlap = 0;
   for (i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementVertices(i, vertices);
      GetNodalValues(i, values, vdim);
      for (j = 0; j < vertices.Size(); j++)
      {
         nval(vertices[j]) += values[j];
         overlap[vertices[j]]++;
      }
   }
   for (i = 0; i < overlap.Size(); i++)
   {
      nval(i) /= overlap[i];
   }
}

void GridFunction::AccumulateAndCountZones(Coefficient &coeff,
                                           AvgType type,
                                           Array<int> &zones_per_vdof)
{
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   // Local interpolation
   Array<int> vdofs;
   Vector vals;
   *this = 0.0;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementVDofs(i, vdofs);
      // Local interpolation of coeff.
      vals.SetSize(vdofs.Size());
      fes->GetFE(i)->Project(coeff, *fes->GetElementTransformation(i), vals);

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++)
      {
         if (type == HARMONIC)
         {
            MFEM_VERIFY(vals[j] != 0.0,
                        "Coefficient has zeros, harmonic avg is undefined!");
            (*this)(vdofs[j]) += 1.0 / vals[j];
         }
         else if (type == ARITHMETIC)
         {
            (*this)(vdofs[j]) += vals[j];
         }
         else { MFEM_ABORT("Not implemented"); }

         zones_per_vdof[vdofs[j]]++;
      }
   }
}

void GridFunction::ComputeMeans(AvgType type, Array<int> &zones_per_vdof)
{
   switch (type)
   {
      case ARITHMETIC:
         for (int i = 0; i < size; i++)
         {
            (*this)(i) /= zones_per_vdof[i];
         }
         break;

      case HARMONIC:
         for (int i = 0; i < size; i++)
         {
            (*this)(i) = zones_per_vdof[i]/(*this)(i);
         }
         break;

      default:
         MFEM_ABORT("invalud AvgType");
   }
}

void GridFunction::ProjectDeltaCoefficient(DeltaCoefficient &delta_coeff,
                                           double &integral)
{
   if (!fes->GetNE())
   {
      integral = 0.0;
      return;
   }

   Mesh *mesh = fes->GetMesh();
   const int dim = mesh->Dimension();
   const double *center = delta_coeff.Center();
   const double *vert = mesh->GetVertex(0);
   double min_dist, dist;
   int v_idx = 0;

   // find the vertex closest to the center of the delta function
   min_dist = Distance(center, vert, dim);
   for (int i = 0; i < mesh->GetNV(); i++)
   {
      vert = mesh->GetVertex(i);
      dist = Distance(center, vert, dim);
      if (dist < min_dist)
      {
         min_dist = dist;
         v_idx = i;
      }
   }

   (*this) = 0.0;
   integral = 0.0;

   if (min_dist >= delta_coeff.Tol())
   {
      return;
   }

   // find the elements that have 'v_idx' as a vertex
   MassIntegrator Mi(*delta_coeff.Weight());
   DenseMatrix loc_mass;
   Array<int> vdofs, vertices;
   Vector vals, loc_mass_vals;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      mesh->GetElementVertices(i, vertices);
      for (int j = 0; j < vertices.Size(); j++)
         if (vertices[j] == v_idx)
         {
            const FiniteElement *fe = fes->GetFE(i);
            Mi.AssembleElementMatrix(*fe, *fes->GetElementTransformation(i),
                                     loc_mass);
            vals.SetSize(fe->GetDof());
            fe->ProjectDelta(j, vals);
            fes->GetElementVDofs(i, vdofs);
            SetSubVector(vdofs, vals);
            loc_mass_vals.SetSize(vals.Size());
            loc_mass.Mult(vals, loc_mass_vals);
            integral += loc_mass_vals.Sum(); // partition of unity basis
            break;
         }
   }
}

void GridFunction::ProjectCoefficient(Coefficient &coeff)
{
   DeltaCoefficient *delta_c = dynamic_cast<DeltaCoefficient *>(&coeff);

   if (delta_c == NULL)
   {
      Array<int> vdofs;
      Vector vals;

      for (int i = 0; i < fes->GetNE(); i++)
      {
         fes->GetElementVDofs(i, vdofs);
         vals.SetSize(vdofs.Size());
         fes->GetFE(i)->Project(coeff, *fes->GetElementTransformation(i), vals);
         SetSubVector(vdofs, vals);
      }
   }
   else
   {
      double integral;

      ProjectDeltaCoefficient(*delta_c, integral);

      (*this) *= (delta_c->Scale() / integral);
   }
}

void GridFunction::ProjectCoefficient(
   Coefficient &coeff, Array<int> &dofs, int vd)
{
   int el = -1;
   ElementTransformation *T = NULL;
   const FiniteElement *fe = NULL;

   for (int i = 0; i < dofs.Size(); i++)
   {
      int dof = dofs[i], j = fes->GetElementForDof(dof);
      if (el != j)
      {
         el = j;
         T = fes->GetElementTransformation(el);
         fe = fes->GetFE(el);
      }
      int vdof = fes->DofToVDof(dof, vd);
      int ld = fes->GetLocalDofForDof(dof);
      const IntegrationPoint &ip = fe->GetNodes().IntPoint(ld);
      T->SetIntPoint(&ip);
      (*this)(vdof) = coeff.Eval(*T, ip);
   }
}

void GridFunction::ProjectCoefficient(VectorCoefficient &vcoeff)
{
   int i;
   Array<int> vdofs;
   Vector vals;

   for (i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementVDofs(i, vdofs);
      vals.SetSize(vdofs.Size());
      fes->GetFE(i)->Project(vcoeff, *fes->GetElementTransformation(i), vals);
      SetSubVector(vdofs, vals);
   }
}

void GridFunction::ProjectCoefficient(
   VectorCoefficient &vcoeff, Array<int> &dofs)
{
   int el = -1;
   ElementTransformation *T = NULL;
   const FiniteElement *fe = NULL;

   Vector val;

   for (int i = 0; i < dofs.Size(); i++)
   {
      int dof = dofs[i], j = fes->GetElementForDof(dof);
      if (el != j)
      {
         el = j;
         T = fes->GetElementTransformation(el);
         fe = fes->GetFE(el);
      }
      int ld = fes->GetLocalDofForDof(dof);
      const IntegrationPoint &ip = fe->GetNodes().IntPoint(ld);
      T->SetIntPoint(&ip);
      vcoeff.Eval(val, *T, ip);
      for (int vd = 0; vd < fes->GetVDim(); vd ++)
      {
         int vdof = fes->DofToVDof(dof, vd);
         (*this)(vdof) = val(vd);
      }
   }
}

void GridFunction::ProjectCoefficient(Coefficient *coeff[])
{
   int i, j, fdof, d, ind, vdim;
   double val;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Array<int> vdofs;

   vdim = fes->GetVDim();
   for (i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      fdof = fe->GetDof();
      transf = fes->GetElementTransformation(i);
      const IntegrationRule &ir = fe->GetNodes();
      fes->GetElementVDofs(i, vdofs);
      for (j = 0; j < fdof; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         transf->SetIntPoint(&ip);
         for (d = 0; d < vdim; d++)
         {
            val = coeff[d]->Eval(*transf, ip);
            if ( (ind = vdofs[fdof*d+j]) < 0 )
            {
               val = -val, ind = -1-ind;
            }
            (*this)(ind) = val;
         }
      }
   }
}

void GridFunction::ProjectDiscCoefficient(VectorCoefficient &coeff,
                                          Array<int> &dof_attr)
{
   Array<int> vdofs;
   Vector vals;

   // maximal element attribute for each dof
   dof_attr.SetSize(fes->GetVSize());
   dof_attr = -1;

   // local projection
   for (int i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementVDofs(i, vdofs);
      vals.SetSize(vdofs.Size());
      fes->GetFE(i)->Project(coeff, *fes->GetElementTransformation(i), vals);

      // the values in shared dofs are determined from the element with maximal
      // attribute
      int attr = fes->GetAttribute(i);
      for (int j = 0; j < vdofs.Size(); j++)
      {
         if (attr > dof_attr[vdofs[j]])
         {
            (*this)(vdofs[j]) = vals[j];
            dof_attr[vdofs[j]] = attr;
         }
      }
   }
}

void GridFunction::ProjectDiscCoefficient(VectorCoefficient &coeff)
{
   Array<int> dof_attr;
   ProjectDiscCoefficient(coeff, dof_attr);
}

void GridFunction::ProjectDiscCoefficient(Coefficient &coeff, AvgType type)
{
   // Harmonic  (x1 ... xn) = [ (1/x1 + ... + 1/xn) / n ]^-1.
   // Arithmetic(x1 ... xn) = (x1 + ... + xn) / n.

   Array<int> zones_per_vdof;
   AccumulateAndCountZones(coeff, type, zones_per_vdof);

   ComputeMeans(type, zones_per_vdof);
}

void GridFunction::ProjectBdrCoefficient(
   Coefficient *coeff[], Array<int> &attr)
{
   int i, j, fdof, d, ind, vdim;
   double val;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Array<int> vdofs;

   vdim = fes->GetVDim();
   for (i = 0; i < fes->GetNBE(); i++)
   {
      if (attr[fes->GetBdrAttribute(i) - 1])
      {
         fe = fes->GetBE(i);
         fdof = fe->GetDof();
         transf = fes->GetBdrElementTransformation(i);
         const IntegrationRule &ir = fe->GetNodes();
         fes->GetBdrElementVDofs(i, vdofs);

         for (j = 0; j < fdof; j++)
         {
            const IntegrationPoint &ip = ir.IntPoint(j);
            transf->SetIntPoint(&ip);
            for (d = 0; d < vdim; d++)
            {
               val = coeff[d]->Eval(*transf, ip);
               if ( (ind = vdofs[fdof*d+j]) < 0 )
               {
                  val = -val, ind = -1-ind;
               }
               (*this)(ind) = val;
            }
         }
      }
   }

   // In the case of partially conforming space, i.e. (fes->cP != NULL), we need
   // to set the values of all dofs on which the dofs set above depend.
   // Dependency is defined from the matrix A = cP.cR: dof i depends on dof j
   // iff A_ij != 0. It is sufficient to resolve just the first level of
   // dependency since A is a projection matrix: A^n = A due to cR.cP = I.
   // Cases like this arise in 3D when boundary edges are constrained by (depend
   // on) internal faces/elements.
   // We use the virtual method GetBoundaryClosure from NCMesh to resolve the
   // dependencies.

   if (fes->Nonconforming() && fes->GetMesh()->Dimension() == 3)
   {
      Vector vals;
      Mesh *mesh = fes->GetMesh();
      NCMesh *ncmesh = mesh->ncmesh;
      Array<int> bdr_edges, bdr_vertices;
      ncmesh->GetBoundaryClosure(attr, bdr_vertices, bdr_edges);

      for (i = 0; i < bdr_edges.Size(); i++)
      {
         int edge = bdr_edges[i];
         fes->GetEdgeVDofs(edge, vdofs);
         if (vdofs.Size() == 0) { continue; }

         transf = mesh->GetEdgeTransformation(edge);
         transf->Attribute = -1; // FIXME: set the boundary attribute
         fe = fes->GetEdgeElement(edge);
         vals.SetSize(fe->GetDof());
         for (d = 0; d < vdim; d++)
         {
            fe->Project(*coeff[d], *transf, vals);
            for (int k = 0; k < vals.Size(); k++)
            {
               (*this)(vdofs[d*vals.Size()+k]) = vals(k);
            }
         }
      }
   }
}

void GridFunction::ProjectBdrCoefficientNormal(
   VectorCoefficient &vcoeff, Array<int> &bdr_attr)
{
#if 0
   // implementation for the case when the face dofs are integrals of the
   // normal component.
   const FiniteElement *fe;
   ElementTransformation *T;
   Array<int> dofs;
   int dim = vcoeff.GetVDim();
   Vector vc(dim), nor(dim), lvec, shape;

   for (int i = 0; i < fes->GetNBE(); i++)
   {
      if (bdr_attr[fes->GetBdrAttribute(i)-1] == 0)
      {
         continue;
      }
      fe = fes->GetBE(i);
      T = fes->GetBdrElementTransformation(i);
      int intorder = 2*fe->GetOrder(); // !!!
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), intorder);
      int nd = fe->GetDof();
      lvec.SetSize(nd);
      shape.SetSize(nd);
      lvec = 0.0;
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         T->SetIntPoint(&ip);
         vcoeff.Eval(vc, *T, ip);
         CalcOrtho(T->Jacobian(), nor);
         fe->CalcShape(ip, shape);
         lvec.Add(ip.weight * (vc * nor), shape);
      }
      fes->GetBdrElementDofs(i, dofs);
      SetSubVector(dofs, lvec);
   }
#else
   // implementation for the case when the face dofs are scaled point
   // values of the normal component.
   const FiniteElement *fe;
   ElementTransformation *T;
   Array<int> dofs;
   int dim = vcoeff.GetVDim();
   Vector vc(dim), nor(dim), lvec;

   for (int i = 0; i < fes->GetNBE(); i++)
   {
      if (bdr_attr[fes->GetBdrAttribute(i)-1] == 0)
      {
         continue;
      }
      fe = fes->GetBE(i);
      T = fes->GetBdrElementTransformation(i);
      const IntegrationRule &ir = fe->GetNodes();
      lvec.SetSize(fe->GetDof());
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         T->SetIntPoint(&ip);
         vcoeff.Eval(vc, *T, ip);
         CalcOrtho(T->Jacobian(), nor);
         lvec(j) = (vc * nor);
      }
      fes->GetBdrElementDofs(i, dofs);
      SetSubVector(dofs, lvec);
   }
#endif
}

void GridFunction::ProjectBdrCoefficientTangent(
   VectorCoefficient &vcoeff, Array<int> &bdr_attr)
{
   const FiniteElement *fe;
   ElementTransformation *T;
   Array<int> dofs;
   Vector lvec;

   for (int i = 0; i < fes->GetNBE(); i++)
   {
      if (bdr_attr[fes->GetBdrAttribute(i)-1] == 0)
      {
         continue;
      }
      fe = fes->GetBE(i);
      T = fes->GetBdrElementTransformation(i);
      fes->GetBdrElementDofs(i, dofs);
      lvec.SetSize(fe->GetDof());
      fe->Project(vcoeff, *T, lvec);
      SetSubVector(dofs, lvec);
   }

   if (fes->Nonconforming() && fes->GetMesh()->Dimension() == 3)
   {
      Mesh *mesh = fes->GetMesh();
      NCMesh *ncmesh = mesh->ncmesh;
      Array<int> bdr_edges, bdr_vertices;
      ncmesh->GetBoundaryClosure(bdr_attr, bdr_vertices, bdr_edges);

      for (int i = 0; i < bdr_edges.Size(); i++)
      {
         int edge = bdr_edges[i];
         fes->GetEdgeDofs(edge, dofs);
         if (dofs.Size() == 0) { continue; }

         T = mesh->GetEdgeTransformation(edge);
         T->Attribute = -1; // FIXME: set the boundary attribute
         fe = fes->GetEdgeElement(edge);
         lvec.SetSize(fe->GetDof());
         fe->Project(vcoeff, *T, lvec);
         SetSubVector(dofs, lvec);
      }
   }
}

double GridFunction::ComputeL2Error(
   Coefficient *exsol[], const IntegrationRule *irs[]) const
{
   double error = 0.0, a;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Vector shape;
   Array<int> vdofs;
   int fdof, d, i, intorder, j, k;

   for (i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      fdof = fe->GetDof();
      transf = fes->GetElementTransformation(i);
      shape.SetSize(fdof);
      intorder = 2*fe->GetOrder() + 1; // <----------
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      fes->GetElementVDofs(i, vdofs);
      for (j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         fe->CalcShape(ip, shape);
         for (d = 0; d < fes->GetVDim(); d++)
         {
            a = 0;
            for (k = 0; k < fdof; k++)
               if (vdofs[fdof*d+k] >= 0)
               {
                  a += (*this)(vdofs[fdof*d+k]) * shape(k);
               }
               else
               {
                  a -= (*this)(-1-vdofs[fdof*d+k]) * shape(k);
               }
            transf->SetIntPoint(&ip);
            a -= exsol[d]->Eval(*transf, ip);
            error += ip.weight * transf->Weight() * a * a;
         }
      }
   }

   if (error < 0.0)
   {
      return -sqrt(-error);
   }
   return sqrt(error);
}

double GridFunction::ComputeL2Error(
   VectorCoefficient &exsol, const IntegrationRule *irs[],
   Array<int> *elems) const
{
   double error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix vals, exact_vals;
   Vector loc_errs;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      if (elems != NULL && (*elems)[i] == 0) { continue; }
      fe = fes->GetFE(i);
      int intorder = 2*fe->GetOrder() + 1; // <----------
      const IntegrationRule *ir;
      /*
       * only for debugging
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      */
      {
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      T = fes->GetElementTransformation(i);
      GetVectorValues(*T, *ir, vals);
      exsol.Eval(exact_vals, *T, *ir);


      // in case of H_DIV_SKEW space we need to transform
      // exact_vals[6] into vals[16] ~ 4x4 skew matrix
      if (fe->GetMapType() == FiniteElement::H_DIV_SKEW)
      {
          DenseMatrix exactmat(4,4); exactmat = 0.0;
          //Vector exact_valsvec(16);
          DenseMatrix exact_valsmat(16, ir->GetNPoints());
          /*
          double t1[4]; Vector t1i(t1, 4);
          double t2[4]; Vector t2i(t2, 4);
          Vector Mt(4);
          Vector v(6);
          Vector exact_dofvec(10);
          */

          for (int j = 0; j < ir->GetNPoints(); j++)
          {
              /*
              exactmat(0,1) =  exact_vals(5,j); exactmat(0,2) = -exact_vals(4,j); exactmat(0,3) =  exact_vals(3,j);
              exactmat(1,0) = -exact_vals(5,j); exactmat(1,2) =  exact_vals(2,j); exactmat(1,3) = -exact_vals(1,j);
              exactmat(2,0) =  exact_vals(4,j); exactmat(2,1) = -exact_vals(2,j); exactmat(2,3) =  exact_vals(0,j);
              exactmat(3,0) = -exact_vals(3,j); exactmat(3,1) =  exact_vals(1,j); exactmat(3,2) = -exact_vals(0,j);
              */

              exactmat(0,1) =  exact_vals(0,j); exactmat(0,2) =  exact_vals(1,j); exactmat(0,3) =  exact_vals(2,j);
              exactmat(1,0) = -exact_vals(0,j); exactmat(1,2) =  exact_vals(3,j); exactmat(1,3) =  exact_vals(4,j);
              exactmat(2,0) = -exact_vals(1,j); exactmat(2,1) = -exact_vals(3,j); exactmat(2,3) =  exact_vals(5,j);
              exactmat(3,0) = -exact_vals(2,j); exactmat(3,1) = -exact_vals(4,j); exactmat(3,2) = -exact_vals(5,j);

              int dim = 4;
              for (int k=0; k<dim; k++)
                 for (int l=0; l<dim; l++)
                 {
                    exact_valsmat(dim*k+l, j) = exactmat(k,l);
                    //exact_valsvec(dim*k+l) = exactmat(k,l);
                 }

              int fdof = fe->GetDof();
              DenseMatrix shape(fdof,dim*dim);

              Array<int> vdofs;
              fes->GetElementVDofs(i, vdofs);

              const IntegrationPoint &ip = ir->IntPoint(j);
              T->SetIntPoint(&ip);
              //const DenseMatrix &J = T->Jacobian();

              fe->CalcVShape(*T, shape);

              Vector valsvec(dim*dim);
              valsvec = 0.0;
              for (int k = 0; k < fdof; k++)
              {
                 if (vdofs[k] >= 0)
                 {
                    for (int l=0; l<dim*dim; l++) { valsvec(l) += shape(k,l)* (*this)(vdofs[k]); }
                 }
                 else
                 {
                    for (int l=0; l<dim*dim; l++) { valsvec(l) -= shape(k,l)*(*this)(-1-vdofs[k]); }
                 }
              }

              if (i == 0 && j == 0)
              {
                  //std::cout << "shape \n";
                  //shape.Print();
                  //std::cout << "vdofs \n";
                  //vdofs.Print();
                  std::cout << "valsvec ~ elsol \n";
                  valsvec.Print();
                  //std::cout << "exact_valsvec ~ exactSolVec \n";
                  //exact_valsvec.Print();
                  //std::cout << "exactmat \n";
                  //exactmat.Print();
                  //std::cout << "exact_valsvec ~ exactSolVec \n";
                  //exact_valsvec.Print();
              }



          }
          //exact_valsmat.Print();
          vals -= exact_valsmat;
      }
      else
        vals -= exact_vals;

      loc_errs.SetSize(vals.Width());
      vals.Norm2(loc_errs);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         error += ip.weight * T->Weight() * (loc_errs(j) * loc_errs(j));
      }
   }

   if (error < 0.0)
   {
      return -sqrt(-error);
   }
   return sqrt(error);
}

double GridFunction::ComputeH1Error(
   Coefficient *exsol, VectorCoefficient *exgrad,
   Coefficient *ell_coeff, double Nu, int norm_type) const
{
   // assuming vdim is 1
   int i, fdof, dim, intorder, j, k;
   Mesh *mesh;
   const FiniteElement *fe;
   ElementTransformation *transf;
   FaceElementTransformations *face_elem_transf;
   Vector e_grad, a_grad, shape, el_dofs, err_val, ell_coeff_val;
   DenseMatrix dshape, dshapet, Jinv;
   Array<int> vdofs;
   IntegrationPoint eip;
   double error = 0.0;

   mesh = fes->GetMesh();
   dim = mesh->Dimension();
   e_grad.SetSize(dim);
   a_grad.SetSize(dim);
   Jinv.SetSize(dim);

   if (norm_type & 1)
      for (i = 0; i < mesh->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         fdof = fe->GetDof();
         transf = mesh->GetElementTransformation(i);
         el_dofs.SetSize(fdof);
         dshape.SetSize(fdof, dim);
         dshapet.SetSize(fdof, dim);
         intorder = 2 * fe->GetOrder(); // <----------
         const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), intorder);
         fes->GetElementVDofs(i, vdofs);
         for (k = 0; k < fdof; k++)
            if (vdofs[k] >= 0)
            {
               el_dofs(k) =   (*this)(vdofs[k]);
            }
            else
            {
               el_dofs(k) = - (*this)(-1-vdofs[k]);
            }
         for (j = 0; j < ir.GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir.IntPoint(j);
            fe->CalcDShape(ip, dshape);
            transf->SetIntPoint(&ip);
            exgrad->Eval(e_grad, *transf, ip);
            CalcInverse(transf->Jacobian(), Jinv);
            Mult(dshape, Jinv, dshapet);
            dshapet.MultTranspose(el_dofs, a_grad);
            e_grad -= a_grad;
            error += (ip.weight * transf->Weight() *
                      ell_coeff->Eval(*transf, ip) *
                      (e_grad * e_grad));
         }
      }

   if (norm_type & 2)
      for (i = 0; i < mesh->GetNFaces(); i++)
      {
         face_elem_transf = mesh->GetFaceElementTransformations(i, 5);
         int i1 = face_elem_transf->Elem1No;
         int i2 = face_elem_transf->Elem2No;
         intorder = fes->GetFE(i1)->GetOrder();
         if (i2 >= 0)
            if ( (k = fes->GetFE(i2)->GetOrder()) > intorder )
            {
               intorder = k;
            }
         intorder = 2 * intorder;  // <-------------
         const IntegrationRule &ir =
            IntRules.Get(face_elem_transf->FaceGeom, intorder);
         err_val.SetSize(ir.GetNPoints());
         ell_coeff_val.SetSize(ir.GetNPoints());
         // side 1
         transf = face_elem_transf->Elem1;
         fe = fes->GetFE(i1);
         fdof = fe->GetDof();
         fes->GetElementVDofs(i1, vdofs);
         shape.SetSize(fdof);
         el_dofs.SetSize(fdof);
         for (k = 0; k < fdof; k++)
            if (vdofs[k] >= 0)
            {
               el_dofs(k) =   (*this)(vdofs[k]);
            }
            else
            {
               el_dofs(k) = - (*this)(-1-vdofs[k]);
            }
         for (j = 0; j < ir.GetNPoints(); j++)
         {
            face_elem_transf->Loc1.Transform(ir.IntPoint(j), eip);
            fe->CalcShape(eip, shape);
            transf->SetIntPoint(&eip);
            ell_coeff_val(j) = ell_coeff->Eval(*transf, eip);
            err_val(j) = exsol->Eval(*transf, eip) - (shape * el_dofs);
         }
         if (i2 >= 0)
         {
            // side 2
            face_elem_transf = mesh->GetFaceElementTransformations(i, 10);
            transf = face_elem_transf->Elem2;
            fe = fes->GetFE(i2);
            fdof = fe->GetDof();
            fes->GetElementVDofs(i2, vdofs);
            shape.SetSize(fdof);
            el_dofs.SetSize(fdof);
            for (k = 0; k < fdof; k++)
               if (vdofs[k] >= 0)
               {
                  el_dofs(k) =   (*this)(vdofs[k]);
               }
               else
               {
                  el_dofs(k) = - (*this)(-1-vdofs[k]);
               }
            for (j = 0; j < ir.GetNPoints(); j++)
            {
               face_elem_transf->Loc2.Transform(ir.IntPoint(j), eip);
               fe->CalcShape(eip, shape);
               transf->SetIntPoint(&eip);
               ell_coeff_val(j) += ell_coeff->Eval(*transf, eip);
               ell_coeff_val(j) *= 0.5;
               err_val(j) -= (exsol->Eval(*transf, eip) - (shape * el_dofs));
            }
         }
         face_elem_transf = mesh->GetFaceElementTransformations(i, 16);
         transf = face_elem_transf->Face;
         for (j = 0; j < ir.GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir.IntPoint(j);
            transf->SetIntPoint(&ip);
            error += (ip.weight * Nu * ell_coeff_val(j) *
                      pow(transf->Weight(), 1.0-1.0/(dim-1)) *
                      err_val(j) * err_val(j));
         }
      }

   if (error < 0.0)
   {
      return -sqrt(-error);
   }
   return sqrt(error);
}

double GridFunction::ComputeMaxError(
   Coefficient *exsol[], const IntegrationRule *irs[]) const
{
   double error = 0.0, a;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Vector shape;
   Array<int> vdofs;
   int fdof, d, i, intorder, j, k;

   for (i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      fdof = fe->GetDof();
      transf = fes->GetElementTransformation(i);
      shape.SetSize(fdof);
      intorder = 2*fe->GetOrder() + 1; // <----------
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      fes->GetElementVDofs(i, vdofs);
      for (j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         fe->CalcShape(ip, shape);
         transf->SetIntPoint(&ip);
         for (d = 0; d < fes->GetVDim(); d++)
         {
            a = 0;
            for (k = 0; k < fdof; k++)
               if (vdofs[fdof*d+k] >= 0)
               {
                  a += (*this)(vdofs[fdof*d+k]) * shape(k);
               }
               else
               {
                  a -= (*this)(-1-vdofs[fdof*d+k]) * shape(k);
               }
            a -= exsol[d]->Eval(*transf, ip);
            a = fabs(a);
            if (error < a)
            {
               error = a;
            }
         }
      }
   }

   return error;
}

double GridFunction::ComputeW11Error(
   Coefficient *exsol, VectorCoefficient *exgrad, int norm_type,
   Array<int> *elems, const IntegrationRule *irs[]) const
{
   // assuming vdim is 1
   int i, fdof, dim, intorder, j, k;
   Mesh *mesh;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Vector e_grad, a_grad, shape, el_dofs, err_val, ell_coeff_val;
   DenseMatrix dshape, dshapet, Jinv;
   Array<int> vdofs;
   double a, error = 0.0;

   mesh = fes->GetMesh();
   dim = mesh->Dimension();
   e_grad.SetSize(dim);
   a_grad.SetSize(dim);
   Jinv.SetSize(dim);

   if (norm_type & 1) // L_1 norm
      for (i = 0; i < mesh->GetNE(); i++)
      {
         if (elems != NULL && (*elems)[i] == 0) { continue; }
         fe = fes->GetFE(i);
         fdof = fe->GetDof();
         transf = fes->GetElementTransformation(i);
         el_dofs.SetSize(fdof);
         shape.SetSize(fdof);
         intorder = 2*fe->GetOrder() + 1; // <----------
         const IntegrationRule *ir;
         if (irs)
         {
            ir = irs[fe->GetGeomType()];
         }
         else
         {
            ir = &(IntRules.Get(fe->GetGeomType(), intorder));
         }
         fes->GetElementVDofs(i, vdofs);
         for (k = 0; k < fdof; k++)
            if (vdofs[k] >= 0)
            {
               el_dofs(k) = (*this)(vdofs[k]);
            }
            else
            {
               el_dofs(k) = -(*this)(-1-vdofs[k]);
            }
         for (j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            fe->CalcShape(ip, shape);
            transf->SetIntPoint(&ip);
            a = (el_dofs * shape) - (exsol->Eval(*transf, ip));
            error += ip.weight * transf->Weight() * fabs(a);
         }
      }

   if (norm_type & 2) // W^1_1 seminorm
      for (i = 0; i < mesh->GetNE(); i++)
      {
         if (elems != NULL && (*elems)[i] == 0) { continue; }
         fe = fes->GetFE(i);
         fdof = fe->GetDof();
         transf = mesh->GetElementTransformation(i);
         el_dofs.SetSize(fdof);
         dshape.SetSize(fdof, dim);
         dshapet.SetSize(fdof, dim);
         intorder = 2*fe->GetOrder() + 1; // <----------
         const IntegrationRule *ir;
         if (irs)
         {
            ir = irs[fe->GetGeomType()];
         }
         else
         {
            ir = &(IntRules.Get(fe->GetGeomType(), intorder));
         }
         fes->GetElementVDofs(i, vdofs);
         for (k = 0; k < fdof; k++)
            if (vdofs[k] >= 0)
            {
               el_dofs(k) = (*this)(vdofs[k]);
            }
            else
            {
               el_dofs(k) = -(*this)(-1-vdofs[k]);
            }
         for (j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            fe->CalcDShape(ip, dshape);
            transf->SetIntPoint(&ip);
            exgrad->Eval(e_grad, *transf, ip);
            CalcInverse(transf->Jacobian(), Jinv);
            Mult(dshape, Jinv, dshapet);
            dshapet.MultTranspose(el_dofs, a_grad);
            e_grad -= a_grad;
            error += ip.weight * transf->Weight() * e_grad.Norml1();
         }
      }

   return error;
}

double GridFunction::ComputeLpError(const double p, Coefficient &exsol,
                                    Coefficient *weight,
                                    const IntegrationRule *irs[]) const
{
   double error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   Vector vals;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         int intorder = 2*fe->GetOrder() + 1; // <----------
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      GetValues(i, *ir, vals);
      T = fes->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         double err = fabs(vals(j) - exsol.Eval(*T, ip));
         if (p < numeric_limits<double>::infinity())
         {
            err = pow(err, p);
            if (weight)
            {
               err *= weight->Eval(*T, ip);
            }
            error += ip.weight * T->Weight() * err;
         }
         else
         {
            if (weight)
            {
               err *= weight->Eval(*T, ip);
            }
            error = std::max(error, err);
         }
      }
   }

   if (p < numeric_limits<double>::infinity())
   {
      // negative quadrature weights may cause the error to be negative
      if (error < 0.)
      {
         error = -pow(-error, 1./p);
      }
      else
      {
         error = pow(error, 1./p);
      }
   }

   return error;
}

double GridFunction::ComputeLpError(const double p, VectorCoefficient &exsol,
                                    Coefficient *weight,
                                    VectorCoefficient *v_weight,
                                    const IntegrationRule *irs[]) const
{
   double error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix vals, exact_vals;
   Vector loc_errs;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         int intorder = 2*fe->GetOrder() + 1; // <----------
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      T = fes->GetElementTransformation(i);
      GetVectorValues(*T, *ir, vals);
      exsol.Eval(exact_vals, *T, *ir);
      vals -= exact_vals;
      loc_errs.SetSize(vals.Width());
      if (!v_weight)
      {
         // compute the lengths of the errors at the integration points
         // thus the vector norm is rotationally invariant
         vals.Norm2(loc_errs);
      }
      else
      {
         v_weight->Eval(exact_vals, *T, *ir);
         // column-wise dot product of the vector error (in vals) and the
         // vector weight (in exact_vals)
         for (int j = 0; j < vals.Width(); j++)
         {
            double err = 0.0;
            for (int d = 0; d < vals.Height(); d++)
            {
               err += vals(d,j)*exact_vals(d,j);
            }
            loc_errs(j) = fabs(err);
         }
      }
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         double err = loc_errs(j);
         if (p < numeric_limits<double>::infinity())
         {
            err = pow(err, p);
            if (weight)
            {
               err *= weight->Eval(*T, ip);
            }
            error += ip.weight * T->Weight() * err;
         }
         else
         {
            if (weight)
            {
               err *= weight->Eval(*T, ip);
            }
            error = std::max(error, err);
         }
      }
   }

   if (p < numeric_limits<double>::infinity())
   {
      // negative quadrature weights may cause the error to be negative
      if (error < 0.)
      {
         error = -pow(-error, 1./p);
      }
      else
      {
         error = pow(error, 1./p);
      }
   }

   return error;
}

GridFunction & GridFunction::operator=(double value)
{
   for (int i = 0; i < size; i++)
   {
      data[i] = value;
   }
   return *this;
}

GridFunction & GridFunction::operator=(const Vector &v)
{
   MFEM_ASSERT(fes && v.Size() == fes->GetVSize(), "");
   SetSize(v.Size());
   for (int i = 0; i < size; i++)
   {
      data[i] = v(i);
   }
   return *this;
}

GridFunction & GridFunction::operator=(const GridFunction &v)
{
   return this->operator=((const Vector &)v);
}

void GridFunction::Save(std::ostream &out) const
{
   fes->Save(out);
   out << '\n';
   if (fes->GetOrdering() == Ordering::byNODES)
   {
      Vector::Print(out, 1);
   }
   else
   {
      Vector::Print(out, fes->GetVDim());
   }
   out.flush();
}

void GridFunction::SaveVTK(std::ostream &out, const std::string &field_name,
                           int ref)
{
   Mesh *mesh = fes->GetMesh();
   RefinedGeometry *RefG;
   Vector val;
   DenseMatrix vval, pmat;
   int vec_dim = VectorDim();

   if (vec_dim == 1)
   {
      // scalar data
      out << "SCALARS " << field_name << " double 1\n"
          << "LOOKUP_TABLE default\n";
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         RefG = GlobGeometryRefiner.Refine(
                   mesh->GetElementBaseGeometry(i), ref, 1);

         GetValues(i, RefG->RefPts, val, pmat);

         for (int j = 0; j < val.Size(); j++)
         {
            out << val(j) << '\n';
         }
      }
   }
   else if (vec_dim == mesh->Dimension())
   {
      // vector data
      out << "VECTORS " << field_name << " double\n";
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         RefG = GlobGeometryRefiner.Refine(
                   mesh->GetElementBaseGeometry(i), ref, 1);

         GetVectorValues(i, RefG->RefPts, vval, pmat);

         for (int j = 0; j < vval.Width(); j++)
         {
            out << vval(0, j) << ' ' << vval(1, j) << ' ';
            if (vval.Height() == 2)
            {
               out << 0.0;
            }
            else
            {
               out << vval(2, j);
            }
            out << '\n';
         }
      }
   }
   else
   {
      // other data: save the components as separate scalars
      for (int vd = 0; vd < vec_dim; vd++)
      {
         out << "SCALARS " << field_name << vd << " double 1\n"
             << "LOOKUP_TABLE default\n";
         for (int i = 0; i < mesh->GetNE(); i++)
         {
            RefG = GlobGeometryRefiner.Refine(
                      mesh->GetElementBaseGeometry(i), ref, 1);

            GetValues(i, RefG->RefPts, val, pmat, vd + 1);

            for (int j = 0; j < val.Size(); j++)
            {
               out << val(j) << '\n';
            }
         }
      }
   }
   out.flush();
}

void GridFunction::SaveSTLTri(std::ostream &out, double p1[], double p2[],
                              double p3[])
{
   double v1[3] = { p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] };
   double v2[3] = { p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2] };
   double n[] = {  v1[1] * v2[2] - v1[2] * v2[1],
                   v1[2] * v2[0] - v1[0] * v2[2],
                   v1[0] * v2[1] - v1[1] * v2[0]
                };
   double rl = 1.0 / sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
   n[0] *= rl; n[1] *= rl; n[2] *= rl;

   out << " facet normal " << n[0] << ' ' << n[1] << ' ' << n[2]
       << "\n  outer loop"
       << "\n   vertex " << p1[0] << ' ' << p1[1] << ' ' << p1[2]
       << "\n   vertex " << p2[0] << ' ' << p2[1] << ' ' << p2[2]
       << "\n   vertex " << p3[0] << ' ' << p3[1] << ' ' << p3[2]
       << "\n  endloop\n endfacet\n";
}

void GridFunction::SaveSTL(std::ostream &out, int TimesToRefine)
{
   Mesh *mesh = fes->GetMesh();

   if (mesh->Dimension() != 2)
   {
      return;
   }

   int i, j, k, l, n;
   DenseMatrix pointmat;
   Vector values;
   RefinedGeometry * RefG;
   double pts[4][3], bbox[3][2];

   out << "solid GridFunction\n";

   bbox[0][0] = bbox[0][1] = bbox[1][0] = bbox[1][1] =
                                             bbox[2][0] = bbox[2][1] = 0.0;
   for (i = 0; i < mesh->GetNE(); i++)
   {
      n = fes->GetFE(i)->GetGeomType();
      RefG = GlobGeometryRefiner.Refine(n, TimesToRefine);
      GetValues(i, RefG->RefPts, values, pointmat);
      Array<int> &RG = RefG->RefGeoms;
      n = Geometries.NumBdr(n);
      for (k = 0; k < RG.Size()/n; k++)
      {
         for (j = 0; j < n; j++)
         {
            l = RG[n*k+j];
            pts[j][0] = pointmat(0,l);
            pts[j][1] = pointmat(1,l);
            pts[j][2] = values(l);
         }

         if (n == 3)
         {
            SaveSTLTri(out, pts[0], pts[1], pts[2]);
         }
         else
         {
            SaveSTLTri(out, pts[0], pts[1], pts[2]);
            SaveSTLTri(out, pts[0], pts[2], pts[3]);
         }
      }

      if (i == 0)
      {
         bbox[0][0] = pointmat(0,0);
         bbox[0][1] = pointmat(0,0);
         bbox[1][0] = pointmat(1,0);
         bbox[1][1] = pointmat(1,0);
         bbox[2][0] = values(0);
         bbox[2][1] = values(0);
      }

      for (j = 0; j < values.Size(); j++)
      {
         if (bbox[0][0] > pointmat(0,j))
         {
            bbox[0][0] = pointmat(0,j);
         }
         if (bbox[0][1] < pointmat(0,j))
         {
            bbox[0][1] = pointmat(0,j);
         }
         if (bbox[1][0] > pointmat(1,j))
         {
            bbox[1][0] = pointmat(1,j);
         }
         if (bbox[1][1] < pointmat(1,j))
         {
            bbox[1][1] = pointmat(1,j);
         }
         if (bbox[2][0] > values(j))
         {
            bbox[2][0] = values(j);
         }
         if (bbox[2][1] < values(j))
         {
            bbox[2][1] = values(j);
         }
      }
   }

   cout << "[xmin,xmax] = [" << bbox[0][0] << ',' << bbox[0][1] << "]\n"
        << "[ymin,ymax] = [" << bbox[1][0] << ',' << bbox[1][1] << "]\n"
        << "[zmin,zmax] = [" << bbox[2][0] << ',' << bbox[2][1] << ']'
        << endl;

   out << "endsolid GridFunction" << endl;
}

std::ostream &operator<<(std::ostream &out, const GridFunction &sol)
{
   sol.Save(out);
   return out;
}


QuadratureFunction::QuadratureFunction(Mesh *mesh, std::istream &in)
{
   const char *msg = "invalid input stream";
   string ident;

   qspace = new QuadratureSpace(mesh, in);
   own_qspace = true;

   in >> ident; MFEM_VERIFY(ident == "VDim:", msg);
   in >> vdim;

   Load(in, vdim*qspace->GetSize());
}

void QuadratureFunction::Save(std::ostream &out) const
{
   qspace->Save(out);
   out << "VDim: " << vdim << '\n'
       << '\n';
   Vector::Print(out, vdim);
   out.flush();
}

std::ostream &operator<<(std::ostream &out, const QuadratureFunction &qf)
{
   qf.Save(out);
   return out;
}


double ZZErrorEstimator(BilinearFormIntegrator &blfi,
                        GridFunction &u,
                        GridFunction &flux, Vector &error_estimates,
                        Array<int>* aniso_flags,
                        int with_subdomains)
{
   const int with_coeff = 0;
   FiniteElementSpace *ufes = u.FESpace();
   FiniteElementSpace *ffes = flux.FESpace();
   ElementTransformation *Transf;

   int dim = ufes->GetMesh()->Dimension();
   int nfe = ufes->GetNE();

   Array<int> udofs;
   Array<int> fdofs;
   Vector ul, fl, fla, d_xyz;

   error_estimates.SetSize(nfe);
   if (aniso_flags)
   {
      aniso_flags->SetSize(nfe);
      d_xyz.SetSize(dim);
   }

   int nsd = 1;
   if (with_subdomains)
   {
      for (int i = 0; i < nfe; i++)
      {
         int attr = ufes->GetAttribute(i);
         if (attr > nsd) { nsd = attr; }
      }
   }

   double total_error = 0.0;
   for (int s = 1; s <= nsd; s++)
   {
      // This calls the parallel version when u is a ParGridFunction
      u.ComputeFlux(blfi, flux, with_coeff, (with_subdomains ? s : -1));

      for (int i = 0; i < nfe; i++)
      {
         if (with_subdomains && ufes->GetAttribute(i) != s) { continue; }

         ufes->GetElementVDofs(i, udofs);
         ffes->GetElementVDofs(i, fdofs);

         u.GetSubVector(udofs, ul);
         flux.GetSubVector(fdofs, fla);

         Transf = ufes->GetElementTransformation(i);
         blfi.ComputeElementFlux(*ufes->GetFE(i), *Transf, ul,
                                 *ffes->GetFE(i), fl, with_coeff);

         fl -= fla;

         double err = blfi.ComputeFluxEnergy(*ffes->GetFE(i), *Transf, fl,
                                             (aniso_flags ? &d_xyz : NULL));

         error_estimates(i) = std::sqrt(err);
         total_error += err;

         if (aniso_flags)
         {
            double sum = 0;
            for (int k = 0; k < dim; k++)
            {
               sum += d_xyz[k];
            }

            double thresh = 0.15 * 3.0/dim;
            int flag = 0;
            for (int k = 0; k < dim; k++)
            {
               if (d_xyz[k] / sum > thresh) { flag |= (1 << k); }
            }

            (*aniso_flags)[i] = flag;
         }
      }
   }

   return std::sqrt(total_error);
}


double ComputeElementLpDistance(double p, int i,
                                GridFunction& gf1, GridFunction& gf2)
{
   double norm = 0.0;

   FiniteElementSpace *fes1 = gf1.FESpace();
   FiniteElementSpace *fes2 = gf2.FESpace();

   const FiniteElement* fe1 = fes1->GetFE(i);
   const FiniteElement* fe2 = fes2->GetFE(i);

   const IntegrationRule *ir;
   int intorder = 2*std::max(fe1->GetOrder(),fe2->GetOrder()) + 1; // <-------
   ir = &(IntRules.Get(fe1->GetGeomType(), intorder));
   int nip = ir->GetNPoints();
   Vector val1, val2;


   ElementTransformation *T = fes1->GetElementTransformation(i);
   for (int j = 0; j < nip; j++)
   {
      const IntegrationPoint &ip = ir->IntPoint(j);
      T->SetIntPoint(&ip);

      gf1.GetVectorValue(i, ip, val1);
      gf2.GetVectorValue(i, ip, val2);

      val1 -= val2;
      double err = val1.Norml2();
      if (p < numeric_limits<double>::infinity())
      {
         err = pow(err, p);
         norm += ip.weight * T->Weight() * err;
      }
      else
      {
         norm = std::max(norm, err);
      }
   }

   if (p < numeric_limits<double>::infinity())
   {
      // Negative quadrature weights may cause the norm to be negative
      if (norm < 0.)
      {
         norm = -pow(-norm, 1./p);
      }
      else
      {
         norm = pow(norm, 1./p);
      }
   }

   return norm;
}


double ExtrudeCoefficient::Eval(ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   ElementTransformation *T_in =
      mesh_in->GetElementTransformation(T.ElementNo / n);
   T_in->SetIntPoint(&ip);
   return sol_in.Eval(*T_in, ip);
}


GridFunction *Extrude1DGridFunction(Mesh *mesh, Mesh *mesh2d,
                                    GridFunction *sol, const int ny)
{
   GridFunction *sol2d;

   FiniteElementCollection *solfec2d;
   const char *name = sol->FESpace()->FEColl()->Name();
   string cname = name;
   if (cname == "Linear")
   {
      solfec2d = new LinearFECollection;
   }
   else if (cname == "Quadratic")
   {
      solfec2d = new QuadraticFECollection;
   }
   else if (cname == "Cubic")
   {
      solfec2d = new CubicFECollection;
   }
   else if (!strncmp(name, "H1_", 3))
   {
      solfec2d = new H1_FECollection(atoi(name + 7), 2);
   }
   else if (!strncmp(name, "H1Pos_", 6))
   {
      // use regular (nodal) H1_FECollection
      solfec2d = new H1_FECollection(atoi(name + 10), 2);
   }
   else if (!strncmp(name, "L2_T", 4))
   {
      solfec2d = new L2_FECollection(atoi(name + 10), 2);
   }
   else if (!strncmp(name, "L2_", 3))
   {
      solfec2d = new L2_FECollection(atoi(name + 7), 2);
   }
   else
   {
      cerr << "Extrude1DGridFunction : unknown FE collection : "
           << cname << endl;
      return NULL;
   }
   FiniteElementSpace *solfes2d;
   // assuming sol is scalar
   solfes2d = new FiniteElementSpace(mesh2d, solfec2d);
   sol2d = new GridFunction(solfes2d);
   sol2d->MakeOwner(solfec2d);
   {
      GridFunctionCoefficient csol(sol);
      ExtrudeCoefficient c2d(mesh, csol, ny);
      sol2d->ProjectCoefficient(c2d);
   }
   return sol2d;
}

// This function is similar to the Mesh::computeSliceCell() but additionally computes the
// values of the grid function in the slice cell vertexes.
// (It is the absolute value for vector finite elements)
// computes number of slice cell vertexes, slice cell vertex indices and coordinates and
// for a given element with index = elind.
// updates the edgemarkers and vertex_count correspondingly
// pvec defines the slice plane
void GridFunction::computeSliceCellValues (int elind, vector<vector<double> > & pvec, vector<vector<double> > & ipoints, vector<int>& edgemarkers,
                             vector<vector<double> >& cellpnts, vector<int>& elvertslocal, int & nip, int & vertex_count, vector<double>& vertvalues)
{
    Mesh * mesh = FESpace()->GetMesh();

    bool verbose = false; // probably should be a function argument
    int dim = mesh->Dimension();

    Array<int> edgev(2);
    double * v1, * v2;

    vector<vector<double> > edgeends(dim);
    edgeends[0].reserve(dim);
    edgeends[1].reserve(dim);

    DenseMatrix M(dim, dim);
    Vector sol(4), rh(4);

    vector<double> ip(dim);

    int edgenolen, edgeind;
    //int * edgeindices;
    //edgeindices = mesh->el_to_edge->GetRow(elind);
    //edgenolen = mesh->el_to_edge->RowSize(elind);
    Array<int> cor; // dummy
    Array<int> edgeindices;
    mesh->GetElementEdges(elind, edgeindices, cor);
    edgenolen = mesh->GetElement(elind)->GetNEdges();

    nip = 0;

    Array<int> vertices;
    mesh->GetElementVertices(elind, vertices);
    double val1, val2;

    double pvalue; // value of the grid function at the middle of the edge
    int permut[2]; // defines which of the edge vertexes is the lowest w.r.t time

    Vector pointval1, pointval2;
    IntegrationPoint integp;
    integp.Init();

    for ( int edgeno = 0; edgeno < edgenolen; ++edgeno)
    {
        // true mesh edge index
        edgeind = edgeindices[edgeno];

        mesh->GetEdgeVertices(edgeind, edgev);

        // vertex coordinates
        v1 = mesh->GetVertex(edgev[0]);
        v2 = mesh->GetVertex(edgev[1]);

        // vertex coordinates as vectors of doubles, edgeends 0 is lower in time coordinate than edgeends[1]
        if (v1[dim-1] < v2[dim-1])
        {
            for ( int coo = 0; coo < dim; ++coo)
            {
                edgeends[0][coo] = v1[coo];
                edgeends[1][coo] = v2[coo];
            }
            permut[0] = 0;
            permut[1] = 1;
        }
        else
        {
            for ( int coo = 0; coo < dim; ++coo)
            {
                edgeends[0][coo] = v2[coo];
                edgeends[1][coo] = v1[coo];
            }
            permut[0] = 1;
            permut[1] = 0;
        }

        for ( int vno = 0; vno < mesh->GetElement(elind)->GetNVertices(); ++vno)
        {
            int vind = vertices[vno];
            if (vno == 0)
            {
                if (dim == 3)
                    integp.Set3(0.0,0.0,0.0);
                else // dim == 4
                    integp.Set4(0.0,0.0,0.0,0.0);
            }
            if (vno == 1)
            {
                if (dim == 3)
                    integp.Set3(1.0,0.0,0.0);
                else // dim == 4
                    integp.Set4(1.0,0.0,0.0,0.0);
            }
            if (vno == 2)
            {
                if (dim == 3)
                    integp.Set3(0.0,1.0,0.0);
                else // dim == 4
                    integp.Set4(0.0,1.0,0.0,0.0);
            }
            if (vno == 3)
            {
                if (dim == 3)
                    integp.Set3(0.0,0.0,1.0);
                else // dim == 4
                    integp.Set4(0.0,0.0,1.0,0.0);
            }
            if (vno == 4)
            {
                integp.Set4(0.0,0.0,0.0,1.0);
            }

            if (edgev[permut[0]] == vind)
                GetVectorValue(elind, integp, pointval1);
            if (edgev[permut[1]] == vind)
                GetVectorValue(elind, integp, pointval2);
        }

        val1 = 0.0; val2 = 0.0;
        for ( int coo = 0; coo < dim; ++coo)
        {
            val1 += pointval1[coo] * pointval1[coo];
            val2 += pointval2[coo] * pointval2[coo];
        }
        //cout << "val1 = " << val1 << " val2 = " << val2 << endl;

        val1 = sqrt (val1); val2 = sqrt (val2);

        if (verbose)
        {
            cout << "vertex 1: val1 = " << val1 << endl;
            /*
            for ( int vno = 0; vno < mesh->Dimension(); ++vno)
                cout << v1[vno] << " ";
            cout << endl;
            */
            cout << "vertex 2: val2 = " << val2 <<  endl;
            /*
            for ( int vno = 0; vno < mesh->Dimension(); ++vno)
                cout << v2[vno] << " ";
            cout << endl;
            */
        }

        if (verbose)
        {
            cout << "edgeind " << edgeind << endl;

            cout << "edge vertices:" << endl;
            for (int i = 0; i < 2; ++i)
            {
                cout << "vert ";
                for ( int coo = 0; coo < dim; ++coo)
                    cout << edgeends[i][coo] << " ";
                cout << "   ";
            }
            cout << endl;
        }

        // creating the matrix for computing the intersection point
        for ( int i = 0; i < dim; ++i)
            for ( int j = 0; j < dim - 1; ++j)
                M(i,j) = pvec[j + 1][i];
        for ( int i = 0; i < dim; ++i)
            M(i,dim - 1) = edgeends[0][i] - edgeends[1][i];

        /*
        cout << "M" << endl;
        M.Print();
        cout << "M.Det = " << M.Det() << endl;
        */

        if ( fabs(M.Det()) > MYZEROTOL )
        {
            M.Invert();

            // setting righthand side
            for ( int i = 0; i < dim; ++i)
                rh[i] = edgeends[0][i] - pvec[0][i];

            // solving the system
            M.Mult(rh, sol);

        }
        else
            if (verbose)
                cout << "Edge is parallel" << endl;

        //val1 = edgeends[0][dim-1]; val2 = edgeends[1][dim-1]; only for debugging: delete this
        pvalue = sol[dim-1] * val1 + (1.0 - sol[dim-1]) * val2;

        if (verbose)
        {
            cout << fixed << setprecision(6);
            cout << "val1 = " << val1 << " val2 = " << val2 << endl;
            cout << "sol = " << sol[dim-1];
            cout << "pvalue = " << pvalue << endl << endl;
            //cout << fixed << setprecision(4);
        }


        if (edgemarkers[edgeind] == -2) // if this edge was not considered
        {
            if ( fabs(M.Det()) > MYZEROTOL )
            {
                if ( sol[dim-1] > 0.0 - MYZEROTOL && sol[dim-1] <= 1.0 + MYZEROTOL)
                {
                    for ( int i = 0; i < dim; ++i)
                        ip[i] = edgeends[0][i] + sol[dim-1] * (edgeends[1][i] - edgeends[0][i]);

                    if (verbose)
                    {
                        cout << "intersection point for this edge: " << endl;
                        for ( int i = 0; i < dim; ++i)
                            cout << ip[i] << " ";
                        cout << endl;
                    }

                    ipoints.push_back(ip);
                    //vrtindices[momentind].push_back(vertex_count);
                    elvertslocal.push_back(vertex_count);
                    vertvalues.push_back(pvalue);
                    edgemarkers[edgeind] = vertex_count;
                    cellpnts.push_back(ip);
                    nip++;
                    vertex_count++;
                }
                else
                {
                    if (verbose)
                        cout << "Line but not edge intersects" << endl;
                    edgemarkers[edgeind] = -1;
                }

            }
            else
                if (verbose)
                    cout << "Edge is parallel" << endl;
        }
        else // the edge was already considered -> edgemarkers store the vertex index
        {
            if (verbose)
                cout << "Edge was already considered" << endl;
            if (edgemarkers[edgeind] >= 0)
            {
                elvertslocal.push_back(edgemarkers[edgeind]);
                vertvalues.push_back(pvalue);
                cellpnts.push_back(ipoints[edgemarkers[edgeind]]);
                nip++;
            }
        }

        if (verbose)
            cout << endl;

        //cout << "tempvec.size = " << tempvec.size() << endl;

    } // end of loop over element edges

    /*
    cout << "vertvalues in the end of slicecompute" << endl;
    for ( int i = 0; i < nip; ++i)
    {
        cout << "vertval = " << vertvalues[i] << endl;
    }
    */

    return;
}

void GridFunction::outputSliceGridFuncVTK ( std::stringstream& fname, std::vector<std::vector<double> > & ipoints,
                                std::list<int> &celltypes, int cellstructsize, std::list<std::vector<int> > &elvrtindices, std::list<double > & cellvalues, bool forvideo)
{
    Mesh * mesh = FESpace()->GetMesh();

    int dim = mesh->Dimension();
    // output in the vtk format for paraview
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);

    ofid << "# vtk DataFile Version 3.0" << endl;
    ofid << "Generated by MFEM" << endl;
    ofid << "ASCII" << endl;
    ofid << "DATASET UNSTRUCTURED_GRID" << endl;

    ofid << "POINTS " << ipoints.size() << " double" << endl;
    for (unsigned int vno = 0; vno < ipoints.size(); ++vno)
    {
        for ( int c = 0; c < dim - 1; ++c )
        {
            ofid << ipoints[vno][c] << " ";
        }
        if (dim == 3)
        {
            if (forvideo == true)
                ofid << 0.0 << " ";
            else
                ofid << ipoints[vno][dim - 1] << " ";
        }
        ofid << endl;
    }

    ofid << "CELLS " << celltypes.size() << " " << cellstructsize << endl;
    std::list<int>::const_iterator iter;
    std::list<vector<int> >::const_iterator iter2;
    for (iter = celltypes.begin(), iter2 = elvrtindices.begin();
         iter != celltypes.end() && iter2 != elvrtindices.end()
         ; ++iter, ++iter2)
    {
        //cout << *it;
        int npoints;
        if (*iter == VTKTETRAHEDRON)
            npoints = 4;
        else if (*iter == VTKWEDGE)
            npoints = 6;
        else if (*iter == VTKQUADRIL)
            npoints = 4;
        else //(*iter == VTKTRIANGLE)
            npoints = 3;
        ofid << npoints << " ";

        for ( int i = 0; i < npoints; ++i)
            ofid << (*iter2)[i] << " ";
        ofid << endl;
    }

    ofid << "CELL_TYPES " << celltypes.size() << endl;
    for (iter = celltypes.begin(); iter != celltypes.end(); ++iter)
    {
        ofid << *iter << endl;
    }


    // cell data
    ofid << "CELL_DATA " << celltypes.size() << endl;
    ofid << "SCALARS cell_scalars double 1" << endl;
    ofid << "LOOKUP_TABLE default" << endl;
    //int cnt = 0;
    std::list<double>::const_iterator iterd;
    for (iterd = cellvalues.begin(); iterd != cellvalues.end(); ++iterd)
    {
        //cout << "cell data: " << *iterd << endl;
        ofid << *iterd << endl;
        //cnt++;
    }
    return;
}

// Computes and outputs in VTK format slice meshes of a given 3D or 4D mesh
// by time-like planes t = t0 + k * deltat, k = 0, ..., Nmoments - 1
// myid is used for creating different output files by different processes
// if the mesh is parallel
// usually it is reasonable to refeer myid to the process id in the communicator
// For each cell, an average of the values of the grid function is computed over
// slice cell vertexes.
void GridFunction::ComputeSlices(double t0, int Nmoments, double deltat, int myid, bool forvideo)
{
    bool verbose = false;

    Mesh * mesh = FESpace()->GetMesh();
    int dim = mesh->Dimension();

    // = -2 if not considered, -1 if considered, but does not intersected, index of this vertex in the new 3d mesh otherwise
    // refilled for each time moment
    vector<int> edgemarkers(mesh->GetNEdges());

    vector<vector<int> > elpartition(mesh->GetNEdges());
    mesh->Compute_elpartition (t0, Nmoments, deltat, elpartition);

    // *************************************************************************
    // step 2 of x: looping over time momemnts and slicing elements for each
    // given time moment, and outputs the resulting slice mesh in VTK format
    // *************************************************************************

    // slicing the elements, time moment over time moment
    int elind;

    vector<vector<double> > pvec(dim);
    for ( int i = 0; i < dim; ++i)
        pvec[i].reserve(dim);

    // output data structures for vtk format
    // for each time moment holds a list with cell type for each cell
    vector<std::list<int> > celltypes(Nmoments);
    // for each time moment holds a list with vertex indices
    //vector<std::list<int>> vrtindices(Nmoments);
    // for each time moment holds a list with cell type for each cell
    vector<std::list<vector<int> > > elvrtindices(Nmoments);
    //vector<std::list<vector<double> > > cellvertvalues(Nmoments); // decided not to use this - don't understand how to output correctly in vtk format afterwards
    vector<std::list<double > > cellvalues(Nmoments);

    // number of integers in cell structure - for each cell 1 integer (number of vertices) +
    // + x integers (vertex indices)
    int cellstructsize;
    int vertex_count; // number of vertices in the slice mesh for a single time moment

    // loop over time moments
    for ( int momentind = 0; momentind < Nmoments; ++momentind )
    {
        if (verbose)
            cout << "Time moment " << momentind << ": time = " << t0 + momentind * deltat << endl;

        // refilling edgemarkers, resetting vertex_count and cellstructsize
        for ( int i = 0; i < mesh->GetNEdges(); ++i)
            edgemarkers[i] = -2;

        vertex_count = 0;
        cellstructsize = 0;

        vector<vector<double> > ipoints;    // one of main arrays: all intersection points for a given time moment
        double cellvalue;                   // averaged cell value computed from vertvalues

        // vectors, defining the plane of the slice p0, p1, p2 (and p3 in 4D)
        // p0 is the time aligned vector for the given time moment
        // p1, p2 (and p3) - basis orts for the plane
        // pvec is {p0,p1,p2,p3} vector
        for ( int i = 0; i < dim; ++i)
            for ( int j = 0; j < dim; ++j)
                pvec[i][dim - 1 - j] = ( i == j ? 1.0 : 0.0);
        pvec[0][dim - 1] = t0 + momentind * deltat;

        // loop over elements intersected by the plane realted to a given time moment
        // here, elno = index in elpartition[momentind]
        for ( unsigned int elno = 0; elno < elpartition[momentind].size(); ++elno)
        //for ( int elno = 0; elno < 2; ++elno)
        {
            vector<int> tempvec;             // vertex indices for the cell of the slice mesh
            tempvec.reserve(6);
            vector<vector<double> > cellpnts; //points of the cell of the slice mesh
            cellpnts.reserve(6);

            vector<double> vertvalues;          // values of the grid function at the nodes of the slice cell

            // true mesh element index
            elind = elpartition[momentind][elno];
            //Element * el = mesh->GetElement(elind);

            if (verbose)
                cout << "Element: " << elind << endl;

            // computing number of intersection points, indices and coordinates for
            // local slice cell vertexes (cellpnts and tempvec)  and adding new intersection
            // points and changing edges markers for a given element elind
            // and plane defined by pvec
            int nip;
            //mesh->computeSliceCell (elind, pvec, ipoints, edgemarkers, cellpnts, tempvec, nip, vertex_count);

            computeSliceCellValues (elind, pvec, ipoints, edgemarkers, cellpnts, tempvec, nip, vertex_count, vertvalues);

            if ( (dim == 4 && (nip != 4 && nip != 6)) || (dim == 3 && (nip != 3 && nip != 4)) )
                cout << "Strange nip =  " << nip << " for elind = " << elind << ", time = " << t0 + momentind * deltat << endl;
            else
            {
                if (nip == 4) // tetrahedron in 3d or quadrilateral in 2d
                    if (dim == 4)
                        celltypes[momentind].push_back(VTKTETRAHEDRON);
                    else // dim == 3
                        celltypes[momentind].push_back(VTKQUADRIL);
                else if (nip == 6) // prism
                    celltypes[momentind].push_back(VTKWEDGE);
                else // nip == 3 = triangle
                    celltypes[momentind].push_back(VTKTRIANGLE);

                cellstructsize += nip + 1;

                elvrtindices[momentind].push_back(tempvec);

                cellvalue = 0.0;
                for ( int i = 0; i < nip; ++i)
                {
                    //cout << "vertval = " << vertvalues[i] << endl;
                    cellvalue += vertvalues[i];
                }
                cellvalue /= nip * 1.0;

                if (verbose)
                    cout << "cellvalue = " << cellvalue << endl;

                //cellvertvalues[momentind].push_back(vertvalues);
                cellvalues[momentind].push_back(cellvalue);

                // special reordering of cell vertices, required for the wedge,
                // tetrahedron and quadrilateral cells
                reorder_cellvertices (dim, nip, cellpnts, elvrtindices[momentind].back());

                if (verbose)
                    cout << "nip for the element = " << nip << endl;
            }

        } // end of loop over elements for a given time moment

        // intermediate output
        std::stringstream fname;
        fname << "slicegridfunc_"<< dim - 1 << "d_myid_" << myid << "_moment_" << momentind << ".vtk";
        //outputSliceGridFuncVTK (fname, ipoints, celltypes[momentind], cellstructsize, elvrtindices[momentind], cellvertvalues[momentind]);
        outputSliceGridFuncVTK (fname, ipoints, celltypes[momentind], cellstructsize, elvrtindices[momentind], cellvalues[momentind], forvideo);


    } //end of loop over time moments

    // if not deleted here, gets segfault for more than two parallel refinements afterwards, but this is for GridFunction
    //delete mesh->edge_vertex;
    //mesh->edge_vertex = NULL;

    //

    return;
}

}
