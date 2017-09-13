
/*
 * This example was created during implementation of local refinement in 4D via bisection
 * */

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

double testfun(const Vector& xt);

int main(int argc, char *argv[])
{
    StopWatch chrono;

    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    bool verbose = (myid == 0);
    bool visualization = 1;

    int nDimensions     = 3;

    int ser_ref_levels  = 0;
    int par_ref_levels  = 1;

    const char *mesh_file = "../data/1tet.mesh";
    //const char *mesh_file = "../data/2tets.mesh";
    //const char *mesh_file = "../data/2tets_another.mesh";
    //const char *mesh_file = "../data/1penta.MFEM";
    //const char *mesh_file = "../data/2penta.MFEM";

    //const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    //const char *mesh_file = "../data/cube4d_low.MFEM";
    //const char *mesh_file = "../data/cube4d.MFEM";
    //const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";q

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
        cout << "Studying local mesh refinement in 4D \n";

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&ser_ref_levels, "-sref", "--sref",
                   "Number of serial refinements 4d mesh.");
    args.AddOption(&par_ref_levels, "-pref", "--pref",
                   "Number of parallel refinements 4d mesh.");
    args.AddOption(&nDimensions, "-dim", "--whichD",
                   "Dimension of the space-time problem.");
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
    else //if nDimensions is no 3 or 4
    {
        if (verbose)
            cerr << "Case nDimensions = " << nDimensions << " is not supported \n"
                 << flush;
        MPI_Finalize();
        return -1;

    }
    int dim = mesh->Dimension();

    MFEM_ASSERT(dim == nDimensions, "nDimensions != mesh dimension \n");

    if (verbose) std::cout << "Input mesh info: \n";
    mesh->PrintInfo(std::cout);
    if(verbose) std::cout << "\n";

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

    Array<int> marked_el;
    marked_el.Append(0);
    //marked_el.Append(1);
    for (int l = 0; l < par_ref_levels; l++)
    {
        //pmesh->UniformRefinement();
        pmesh->GeneralRefinement(marked_el);
    }

    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;


    FiniteElementCollection *h1_coll, *l2_coll;
    if (dim == 4)
    {
        if(feorder <= 1)
        {
            h1_coll = new LinearFECollection;
            if (verbose)cout << "H1: order 1 for 4D" << endl;
        }
        else
        {
            h1_coll = new QuadraticFECollection;
            if (verbose)cout << "H1: order 2 for 4D" << endl;
        }

        l2_coll = new L2_FECollection(0, dim);
        if (verbose)cout << "L2: order 0 for 4D" << endl;
    }
    else
    {
        h1_coll = new H1_FECollection(feorder+1, dim);
        if (verbose)cout << "H1: order " << feorder + 1 << " for 3D" << endl;
        l2_coll = new L2_FECollection(feorder, dim);
        if (verbose)cout << "L2: order " << feorder << " for 3D" << endl;
    }

    FunctionCoefficient testfun_coeff(testfun);

    ParFiniteElementSpace *H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);
    ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

    ParGridFunction * testfun_Hh = new ParGridFunction(H_space);
    testfun_Hh->ProjectCoefficient(testfun_coeff);

    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream u_sock(vishost, visport);
        u_sock << "parallel " << num_procs << " " << myid << "\n";
        u_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        u_sock << "solution\n" << *pmesh << *testfun_Hh
               << "window_title 'testfun_Hh'" << endl;
    }

    delete H_space;
    delete W_space;

    MPI_Finalize();

    return 0;
}

double testfun(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z(0.0);
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);
    return x + y + z + t;
}
