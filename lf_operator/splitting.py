"""Input-only Latin-hypercube-guided allocation of existing discrete cases."""
import numpy as np
from pathlib import Path

DENSE_CASE_LIMIT=2048


def case_parameter_inputs(cache,records):
    """Read physical inputs only; the four-dimensional design needs uniform profiles."""
    parameters=[]
    for record in records:
        with np.load(Path(cache)/record['file'],allow_pickle=False) as saved:
            x,a=saved['x'],saved['a']
            if x.ndim!=3 or x.shape[:2]!=(20,2) or a.shape!=(2,):
                raise ValueError('expected parameter-only inputs [20,2,N] and [U,TSR]')
            if not np.allclose(x,x[:1,:,:1]):
                raise ValueError('four-dimensional LHS split requires uniform chord/thickness profiles')
            parameters.append(np.concatenate((a,x[0,:,0])))
    return np.asarray(parameters,dtype=float)


def latin_hypercube_split(inputs,seed=42,val_fraction=.15,test_fraction=.15):
    """Match independent subset LHS designs jointly to unique observed cases.

    Existing cases are not a freely sampled design: exact Latin strata for the
    assigned observations cannot be guaranteed. No outputs enter the matching.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import qmc
    x=np.asarray(inputs,dtype=float)
    if x.ndim!=2 or len(x)<3 or x.shape[1]<1 or not np.isfinite(x).all():
        raise ValueError('split inputs must be finite [N,D] with at least three cases')
    if not 0<val_fraction<1 or not 0<test_fraction<1 or val_fraction+test_fraction>=1:
        raise ValueError('invalid split fractions')
    n=len(x);nv=max(1,round(n*val_fraction));nt=max(1,round(n*test_fraction))
    if nv+nt>=n:raise ValueError('split leaves no training cases')
    counts=dict(train=n-nv-nt,val=nv,test=nt)
    span=np.ptp(x,axis=0);active=span>0
    normalized=(x-x.min(0))/np.where(active,span,1.)
    seeds=np.random.SeedSequence(seed).generate_state(3)
    design={name:qmc.LatinHypercube(d=x.shape[1],seed=int(s)).random(size)
            for (name,size),s in zip(counts.items(),seeds)}
    targets=np.concatenate(list(design.values()))
    if n<=DENSE_CASE_LIMIT:
        cost=np.sum((targets[:,None,active]-normalized[None,:,active])**2,axis=-1)
        rows,columns=linear_sum_assignment(cost)
        assignment=np.empty(n,dtype=int);assignment[rows]=columns
        distance=float(cost[rows,columns].mean());solver='dense_hungarian';neighbors=n
    elif not active.any():
        assignment=np.arange(n);distance=0.;solver='constant_inputs';neighbors=0
    else:
        # Avoid an N x N x D temporary for thousands of LF cases. The solution
        # is optimal on the candidate graph, not necessarily over all pairs.
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import min_weight_full_bipartite_matching
        tree=cKDTree(normalized[:,active]);neighbors=min(64,n)
        while True:
            distances,columns=tree.query(targets[:,active],k=neighbors)
            graph=csr_matrix(((distances**2+1e-12).ravel(),
                (np.repeat(np.arange(n),neighbors),columns.ravel())),shape=(n,n))
            try:
                rows,matched=min_weight_full_bipartite_matching(graph)
                assignment=np.empty(n,dtype=int);assignment[rows]=matched
                distance=float(np.mean(np.sum((targets[:,active]-normalized[assignment][:,active])**2,axis=1)))
                break
            except ValueError:
                if neighbors==n:raise
                neighbors=min(n,neighbors*2)
        solver='sparse_candidate_min_cost'
    splits={};start=0
    for name,size in counts.items():
        splits[name]=sorted(assignment[start:start+size].tolist());start+=size
    report=dict(method='latin_hypercube',seed=seed,counts=counts,input_min=x.min(0).tolist(),
                input_max=x.max(0).tolist(),active_dimensions=active.tolist(),
                unit_design={k:v.tolist() for k,v in design.items()},
                matched_indices_in_design_order=assignment.tolist(),
                mean_squared_assignment_distance=distance,assignment_solver=solver,candidate_neighbors=neighbors,
                scope='Approximate LHS coverage by unique matching of existing cases; not geometry held out')
    return splits,report
