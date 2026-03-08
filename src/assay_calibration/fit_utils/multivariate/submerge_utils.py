def submerge():
    raise NotImplementedError('submerge is deprecated')
    if submerge_steps is None:
    else:
        # submerge mode must rewind; for loop is not suitable
        submerge_mode = False
        i = 0

        history.append(
            dict(component_params=updated_component_params, weights=updated_weights)
        )

        last_surface_idx = -1
        while i < MAX_EM_ITERS:
            if np.isnan(likelihoods).any():
                raise ValueError()
            if np.isnan(np.concatenate(updated_component_params)).any():
                raise ValueError()
            if np.isnan(updated_weights).any():
                raise ValueError()
            if np.isnan(np.concatenate(updated_component_params)).any():
                raise ValueError(
                    f"NaN in updated component params at iteration {i}\n{updated_component_params}"
                )
            if np.isnan(updated_weights).any():
                raise ValueError(
                    f"NaN in updated weights at iteration {i}\n{updated_weights}"
                )

            if not submerge_mode:
                print(f"Before em_iteration {i}:")
                print(f"  Params: {updated_component_params}")
            updated_component_params, updated_weights = em_iteration(
                observations,
                sample_indicators,
                updated_component_params,
                updated_weights,
                constrained=not submerge_mode,
                xlims=xlims,
                iterNum=i + 1
            )
            
            if not submerge_mode:
                # save only last run with constraint for later rewind
                history.append(  # i+1 will be index of history dict (because of initial params)
                    dict(component_params=updated_component_params, weights=updated_weights)
                )
                last_surface_idx = i
            
            if not submerge_mode:
                print(f"After em_iteration {i}:")  
                print(f"  Params: {updated_component_params}\n")


            # if not submerge mode or came back up for air
            if not submerge_mode or (submerge_mode and not constraints.multicomponent_density_constraint_violated(updated_component_params, xlims)):
                # in exploitation mode; continue as normal

                if submerge_mode:
                    # came out of water; append latest history
                    print(f'constraint satisfied: came out of water from i={i} to i={len(history)-2}')
                    submerge_mode = False
                    i = last_surface_idx+1
                    history.append(  # i+1 will be index of history dict (because of initial params)
                        dict(component_params=updated_component_params, weights=updated_weights)
                    )

                print('iteration',i)
                    
            
                likelihoods = np.array(
                    [
                        *likelihoods,
                        get_likelihood(
                            observations,
                            sample_indicators,
                            updated_component_params,
                            updated_weights,
                        )
                        / len(sample_indicators),
                    ]
                )
                if i > 0 and (likelihoods[-1] < likelihoods[-2]):
                    decrease = likelihoods[-2] - likelihoods[-1]
                    
                    relative_decrease = decrease / abs(likelihoods[-2])
                    
                    is_numerical_error = relative_decrease < 1e-6
                    
                    if is_numerical_error:
                        print(f"Iteration {i}: Likelihood decreased by {decrease:.2e} (relative: {relative_decrease:.2e}) - likely numerical rounding error")
                    else:
                        print(f"Iteration {i}: Likelihood DECREASED by {decrease:.2e} (relative: {relative_decrease:.2e}) - algorithmic issue")
                    
                    # Continue if it's just numerical error
                    if is_numerical_error:
                        # Small enough to ignore, continue iteration
                        pass
                    else:
                        # Return failed fit
                        return dict(
                            component_params=updated_component_params,  # Fixed: was [[]*N_components]
                            weights=updated_weights,
                            likelihoods=[*likelihoods, -1 * np.inf],
                            kmeans=kmeans,
                        )
                    # raise ValueError(
                    #     f"Likelihood decreased at iteration {i} for unconstrained fit"
                    # )
                if kwargs.get("verbose", True):
                    pbar.set_postfix({"likelihood": f"{likelihoods[-1]:.6f}"})  # type: ignore
                    pbar.update(1)  # type: ignore
                if (
                    kwargs.get("early_stopping", True)
                    and i >= 1
                    and (np.abs(likelihoods[-1] - likelihoods[-2]) < 1e-10).all()
                ):
                    print('early stopping')
                    break

                submerge_mode = True # go back underwater
            else:
                # in exploration mode; constraint was violated
                if i - last_surface_idx >= submerge_steps:
                    # rewind back to last time constraint was satisfied. perform another constrained iteration, then go back underwater
                    print(f'after {submerge_steps} steps underwater: constraint not satisfied, rewinding from i={i} to i={last_surface_idx} (len(history)={len(history)})')
                    submerge_mode = False
                    i = last_surface_idx # i += 1 at end
                    # updated_component_params, updated_weights = history[-1]['component_params'], history[-1]['weights']
                    # fixed_underwater_params = fix_to_satisfy_density_constraint(updated_component_params, xlims)
                    # if len(fixed_underwater_params) == 0 or any(len(p) == 0 for p in fixed_underwater_params):
                    updated_component_params, updated_weights = history[-1]['component_params'], history[-1]['weights']
                        # print('could not fix underwater params to density constraint')
                    # else:
                    #     updated_component_params = fixed_underwater_params
                    #     print('fixed underwater params to density constraint')

                    if submerge_steps == 1:
                        submerge_steps = 0
                    else:
                        submerge_steps = submerge_steps // 2 # half on each attempt
                    
            
            i += 1
    
    history.append(
        dict(component_params=updated_component_params, weights=updated_weights)
    )