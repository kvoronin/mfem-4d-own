// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Implementation of class Pentatope


#include "mesh_headers.hpp"

namespace mfem
{


Pentatope::Pentatope(const int *ind, int attr)
   : Element(Geometry::PENTATOPE)
{
   attribute = attr;
   for (int i = 0; i < 5; i++)
   {
      indices[i] = ind[i];
   }

   transform = 0;
}

Pentatope::Pentatope(int ind1, int ind2, int ind3, int ind4, int ind5, int attr)
   : Element(Geometry::PENTATOPE)
{
   attribute  = attr;
   indices[0] = ind1;
   indices[1] = ind2;
   indices[2] = ind3;
   indices[3] = ind4;
   indices[4] = ind5;

   transform = 0;
}

void Pentatope::GetVertices(Array<int> &v) const
{
   v.SetSize(5);
   for (int i = 0; i < 5; i++)
   {
      v[i] = indices[i];
   }
}

void Pentatope::SetVertices(const int *ind)
{
   for (int i = 0; i < 5; i++)
   {
      indices[i] = ind[i];
   }
}

// static method
void Pentatope::GetPointMatrix(unsigned transform, DenseMatrix &pm)
{
   double *a = &pm(0,0), *b = &pm(0,1), *c = &pm(0,2), *d = &pm(0,3), *e = &pm(0,4);

   // initialize to identity
   a[0] = 0.0, a[1] = 0.0, a[2] = 0.0; a[3] = 0.0;
   b[0] = 1.0, b[1] = 0.0, b[2] = 0.0; b[3] = 0.0;
   c[0] = 0.0, c[1] = 1.0, c[2] = 0.0; c[3] = 0.0;
   d[0] = 0.0, d[1] = 0.0, d[2] = 1.0; d[3] = 0.0;
   e[0] = 0.0, e[1] = 0.0, e[2] = 0.0; e[3] = 1.0;

   std::cout << "transform = " << transform << "\n";

   int chain[12], n = 0;
   while (transform)
   {
      chain[n++] = (transform & 7) - 1;
      transform >>= 3;
   }

   std::cout << "n = " << n << "\n";
   std::cout << "Implementation of Pentatope::GetPointMatrix() is not ready \n";
   ;
   return;
}

Element *Pentatope::Duplicate(Mesh *m) const
{
   Pentatope *pent = new Pentatope;
   pent->SetVertices(indices);
   pent->SetAttribute(attribute);
   return pent;
}

Linear4DFiniteElement PentatopeFE;

}
