import pyomo.environ as pyo

print("Checking available solvers...")
solvers = ['gurobi', 'highs']

for solver_name in solvers:
    try:
        solver = pyo.SolverFactory(solver_name)
        available = solver.available()
        print(f"{solver_name}: {available}")
    except Exception as e:
        print(f"{solver_name}: Error - {e}")

print("\nChecking if any solver works with a small test problem...")

# Create a small test problem
model = pyo.ConcreteModel()
model.x = pyo.Var(domain=pyo.Binary)
model.objective = pyo.Objective(expr=model.x, sense=pyo.maximize)

for solver_name in solvers:
    try:
        solver = pyo.SolverFactory(solver_name)
        if solver.available():
            results = solver.solve(model, tee=False)
            if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                print(f"✓ {solver_name}: Successfully solved test problem!")
                break
    except Exception as e:
        continue
else:
    print("✗ No working solver found")
