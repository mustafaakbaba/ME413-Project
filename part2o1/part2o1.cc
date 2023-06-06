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
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <map>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
 
#include <deal.II/base/logstream.h>
 
using namespace dealii;
 
 
template <int dim>
class part2o1
{
public:
  part2o1();
  void run();
 
private:
  void make_grid();
  void setup_system();
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
 
float U = 1; 
template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
    //if((std::pow(p(0), 2.0)+std::pow(p(1), 2.0)-std::pow(0.2, 2.0)) <1e-6 ) return 0;
    if(abs(p(0)-5) < 0.1 ) return U*p(1);
    else if(abs(p(0)+5) < 0.1 ) return U*p(1);
    else if(abs(p(1)-5) < 0.1 ) return U*p(1);
    else if(abs(p(1)+5) < 0.1 ) return U*p(1);
    
    return 0;
}
 
 
 
 
 
 
 
template <int dim>
part2o1<dim>::part2o1()
  : fe(1)
  , dof_handler(triangulation)
{}
 
  


 
template <int dim>
void part2o1<dim>::make_grid()
{
    GridIn<dim> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream input_file("cylinderf.msh");
  Assert(dim == 2, ExcInternalError());
  
 
  grid_in.read_msh(input_file);
  triangulation.refine_global(0);
  /*const SphericalManifold<dim> boundary;
  triangulation.set_all_manifold_ids_on_boundary(0);
  triangulation.set_manifold(0, boundary);*/
  
  std::ofstream out("grid_cylinder.svg");
  GridOut gridout;
  gridout.write_svg(triangulation, out);
 

}
 
 
template <int dim>
void part2o1<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
 
  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
 
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
 
  system_matrix.reinit(sparsity_pattern);
 
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}
 
 
 
template <int dim>
void part2o1<dim>::assemble_system()
{
  QGauss<dim> quadrature_formula(fe.degree + 1);
 
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
                            0 *        // f(x_q)
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
void part2o1<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
 
  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
}
 
 


 
 
float rho = 1.2; //density of air
float p_inf = 101.325*1000;
float velocitysq = 1;
float vel_x = 0; float vel_y = 0;
float velocity_infsq = 1;
 int ii = 0;
template <int dim>
void part2o1<dim>::output_results() const
{
  DataOut<dim> data_out;
 
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
 
  data_out.build_patches();
 
  std::ofstream output("solution.vtk");
  data_out.write_vtk(output);
  
  //calculate pressure  
  DataOut<dim> pressure_data;
  Vector<float> speed(triangulation.n_active_cells());
  Vector<float> pressure(triangulation.n_active_cells());
  DerivativeApproximation::approximate_gradient(dof_handler, solution, speed, 0);
  
  while(ii<triangulation.n_active_cells()) {
    pressure[ii] = rho/2*velocity_infsq - rho/2*std::pow(speed[ii],2);
    ii++;
  }
  
  pressure_data.attach_dof_handler(dof_handler);
  pressure_data.add_data_vector(pressure, "pressure");
  pressure_data.build_patches();
  std::ofstream output2("pressure.vtu");
  pressure_data.write_vtu(output2);
  
  //std::cout << "grad " << triangulation.n_active_cells()     << std::endl;
  
}
 
 
 
 
template <int dim>
void part2o1<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;
 
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}
 
 
 
int main()
{
  
    part2o1<2> laplace_problem_2d;
    laplace_problem_2d.run();
    //grid_1();
 

 
  return 0;
}

