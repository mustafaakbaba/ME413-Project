/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 
 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 */
 
 
 
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
 #include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <deal.II/base/timer.h>
#include <deal.II/base/logstream.h>
#include <deal.II/dofs/dof_renumbering.h> 
 
using namespace dealii;
 
 
template <int dim>
class Part1
{
public:
  Part1();
  void run();
 
private:
  void make_grid();
  void setup_system();
  void setup_system_renumber();
  void assemble_system();
  void solve();
  void output_results() const;
 
  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;
 
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
 
  Vector<double> solution;
  Vector<double> system_rhs;
};
 
 
 
template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};
 
 
 
template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};
 
template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
  double return_value = 0.0;
  for (unsigned int i = 0; i < dim; ++i){
    return_value += std::pow(p(i), 2.0);
            }
    //    std::cout << "   asd: " << dim << std::endl;

  return return_value*atan2(p(1),p(0));
  //
}
 
 
template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
    if((std::pow(p(0), 2.0)+std::pow(p(1), 2.0)-std::pow(0.25, 2.0)) <1e-2 ) return 50;
    if((std::pow(p(0), 2.0)+std::pow(p(1), 2.0)-std::pow(2, 2.0)) < 1e-2 ) return 0;
    return 0;
}
 
 
 
 
 
 
 
template <int dim>
Part1<dim>::Part1()
  : fe(1)
  , dof_handler(triangulation)
{}
 
 
 
template <int dim>
void Part1<dim>::make_grid()
{
  const Point<2> center(0, 0);
  const double   inner_radius = 0.25, outer_radius = 2.0;
  const double rot_angle = 3.14159265358979323846/2;
  GridGenerator::half_hyper_shell(
    triangulation, center, inner_radius, outer_radius, 10); //change the last parameter for adjusting the mesh
  triangulation.refine_global(4); //change the last parameter for adjusting the mesh
  GridTools::rotate(rot_angle, triangulation); 	
  
  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
            
  std::ofstream out("grid1.svg");
  GridOut gridout;
  gridout.write_svg(triangulation, out);
}
 
 
template <int dim>
void Part1<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
 
  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
 
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
  
  std::ofstream out("sparsity-pattern-1.svg");
  sparsity_pattern.print_svg(out);

 
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void Part1<dim>::setup_system_renumber()
{
  dof_handler.distribute_dofs(fe);
 
  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
  DoFRenumbering::Cuthill_McKee(dof_handler);
 
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
  
  std::ofstream out("sparsity-pattern-2.svg");
  sparsity_pattern.print_svg(out);

 
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}
 
 
 
template <int dim>
void Part1<dim>::assemble_system()
{
  QGauss<dim> quadrature_formula(fe.degree+1); //adapt order of gauss quadrature formula
 
 
  RightHandSide<dim> right_hand_side;
 
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
 
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
 
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
 
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
 
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;
 
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx
 
            const auto &x_q = fe_values.quadrature_point(q_index);
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            right_hand_side.value(x_q) *        // f(x_q)
                            fe_values.JxW(q_index));            // dx
          }
 
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
 
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
 
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           BoundaryValues<dim>(),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}
 
 
 
template <int dim>
void Part1<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
 
  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
}
 
 
 
template <int dim>
void Part1<dim>::output_results() const
{
  DataOut<dim> data_out;
 
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
 
  data_out.build_patches();
 
  std::ofstream output(dim == 2 ? "solution-2d.vtk" : "solution-3d.vtk");
  data_out.write_vtk(output);
}
 
 
 
 
template <int dim>
void Part1<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;
 
  make_grid();
  setup_system();
  //setup_system_renumber();//uncomment the above line or this line in order to have reordering matrices
  assemble_system();
  solve();
  output_results();
}
 
 
 
int main()
{
  Timer timer; // creating a timer also starts it
  Part1<2> laplace_problem_2d;
  laplace_problem_2d.run();
  timer.stop();
 
  std::cout << "Elapsed CPU time: " << timer.cpu_time() << " seconds.\n";
  std::cout << "Elapsed wall time: " << timer.wall_time() << " seconds.\n";
   
  // reset timer for the next thing it shall do
  timer.reset();
   

 
  return 0;
}

