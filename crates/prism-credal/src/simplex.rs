use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(crate) type Row = Vec<BigInt>;

pub(crate) struct Tableau {
    rows: Vec<Row>,
    basis: Vec<usize>,
    variables: usize,
}

pub(crate) enum Feasibility {
    Feasible(Tableau),
    Infeasible(Vec<BigRational>),
}

pub(crate) struct Optimum {
    pub value: BigRational,
    pub vertex: Vec<BigRational>,
}

// Replace a row by a positive multiple of itself minus a multiple of the
// pivot row. Dividing out the gcd controls coefficient growth without
// changing the represented equality. The objective has its own z column.
fn eliminate(row: &mut Row, pivot: &Row, column: usize) {
    if row[column].is_zero() {
        return;
    }
    let factor = row[column].clone();
    let entry = &pivot[column];
    for (v, p) in row.iter_mut().zip(pivot) {
        *v = &*v * entry - &factor * p;
        if entry.is_negative() {
            *v = -&*v;
        }
    }
    let gcd = row.iter().fold(BigInt::zero(), |gcd, v| gcd.gcd(v));
    if gcd > BigInt::one() {
        for v in row {
            *v /= &gcd;
        }
    }
}

impl Tableau {
    fn pivot(&mut self, row: usize, col: usize, mut objective: Option<&mut Row>) {
        let pivot = self.rows[row].clone();
        for (i, other) in self.rows.iter_mut().enumerate() {
            if i != row {
                eliminate(other, &pivot, col);
            }
        }
        if let Some(obj) = objective.as_mut() {
            eliminate(obj, &pivot, col);
        }
        if pivot[col].is_negative() {
            for v in &mut self.rows[row] {
                *v = -&*v;
            }
        }
        self.basis[row] = col;
    }

    fn optimize(&mut self, costs: &[BigInt]) -> Result<Row, &'static str> {
        let rhs = self.variables + 1;
        let mut objective: Row = costs.iter().map(|c| -c).collect();
        objective.extend([BigInt::one(), BigInt::zero()]);
        for (row, &basic) in self.rows.iter().zip(&self.basis) {
            eliminate(&mut objective, row, basic);
        }
        // Bland's rule resolves both entering-variable choice and ratio ties.
        for _ in 0..20_000 {
            let Some(col) = objective[..self.variables]
                .iter()
                .position(Signed::is_negative)
            else {
                return Ok(objective);
            };
            let row = self
                .rows
                .iter()
                .enumerate()
                .filter(|(_, row)| row[col].is_positive())
                .min_by(|(i, a), (j, b)| {
                    (&a[rhs] * &b[col])
                        .cmp(&(&b[rhs] * &a[col]))
                        .then_with(|| self.basis[*i].cmp(&self.basis[*j]))
                })
                .map(|(i, _)| i)
                .ok_or("unbounded probability objective")?;
            self.pivot(row, col, Some(&mut objective));
        }
        Err("simplex iteration limit")
    }

    pub fn feasible(matrix: &[Row]) -> Result<Feasibility, &'static str> {
        let n = matrix[0].len() - 1;
        let variables = n + matrix.len();
        let mut tableau = Self {
            rows: Vec::new(),
            basis: (n..variables).collect(),
            variables,
        };
        for (i, input) in matrix.iter().enumerate() {
            let mut row = vec![BigInt::zero(); variables + 2];
            row[..n].clone_from_slice(&input[..n]);
            row[variables + 1] = input[n].clone();
            if input[n].is_negative() {
                for v in &mut row {
                    *v = -&*v;
                }
            }
            row[n + i] = BigInt::one();
            tableau.rows.push(row);
        }
        let mut phase_one = vec![BigInt::zero(); variables];
        phase_one[n..].fill(-BigInt::one());
        let objective = tableau.optimize(&phase_one)?;
        if !objective[variables + 1].is_zero() {
            let scale = &objective[variables];
            let y = matrix
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    let value = &objective[n + i] - scale;
                    let value = if row[n].is_negative() { -value } else { value };
                    BigRational::new(value, scale.clone())
                })
                .collect::<Vec<_>>();
            if !verifies_farkas(matrix, &y) {
                return Err("phase-one certificate failed verification");
            }
            return Ok(Feasibility::Infeasible(y));
        }

        // A zero-valued artificial variable must leave the basis before it can
        // drift positive. If no structural column can replace it, drop the
        // redundant equality. A negative pivot is safe at zero RHS.
        let mut i = 0;
        while i < tableau.rows.len() {
            if tableau.basis[i] >= n {
                let structural = &tableau.rows[i][..n];
                let replacement = structural
                    .iter()
                    .position(Signed::is_positive)
                    .or_else(|| structural.iter().position(|x| !x.is_zero()));
                if let Some(col) = replacement {
                    tableau.pivot(i, col, None);
                } else {
                    tableau.rows.remove(i);
                    tableau.basis.remove(i);
                    continue;
                }
            }
            i += 1;
        }
        for row in &mut tableau.rows {
            row.drain(n..variables);
        }
        tableau.variables = n;
        Ok(Feasibility::Feasible(tableau))
    }

    pub fn maximize(&mut self, costs: &[BigInt]) -> Result<Optimum, &'static str> {
        let z = self.variables;
        let row = self.optimize(costs)?;
        let value = BigRational::new(row[z + 1].clone(), row[z].clone());
        let mut vertex = vec![BigRational::zero(); z];
        for (row, &basic) in self.rows.iter().zip(&self.basis) {
            vertex[basic] = BigRational::new(row[z + 1].clone(), row[basic].clone());
        }
        Ok(Optimum { value, vertex })
    }

    pub fn condition(mut self, evidence: &[BigInt]) -> Result<Self, &'static str> {
        let t = self.variables;
        // For u = w/E(w), t = 1/E(w), replace A w = b by A u - b t = 0.
        for row in &mut self.rows {
            let b = row[t + 1].clone();
            row[t] = -b;
            row[t + 1] = BigInt::zero();
            row.push(BigInt::zero());
        }
        // Add E(u) = 1 and express it in the current basis. Its t
        // coefficient is positive when the current vertex supports E.
        let mut normalization = evidence.to_vec();
        normalization.extend([BigInt::zero(), BigInt::zero(), BigInt::one()]);
        for (row, &basic) in self.rows.iter().zip(&self.basis) {
            eliminate(&mut normalization, row, basic);
        }
        if !normalization[t].is_positive() {
            return Err("conditioning requires a vertex with positive evidence support");
        }
        self.variables += 1;
        self.basis.push(t);
        self.rows.push(normalization);
        self.pivot(self.rows.len() - 1, t, None);
        Ok(self)
    }
}

pub(crate) fn verifies_farkas(matrix: &[Row], y: &[BigRational]) -> bool {
    if y.len() != matrix.len() {
        return false;
    }
    let column_sum = |j: usize| {
        matrix
            .iter()
            .zip(y)
            .filter(|(row, v)| !v.is_zero() && !row[j].is_zero())
            .map(|(row, v)| v * &row[j])
            .sum::<BigRational>()
    };
    let rhs = matrix[0].len() - 1;
    column_sum(rhs).is_negative() && (0..rhs).all(|j| !column_sum(j).is_negative())
}
