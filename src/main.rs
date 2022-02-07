//! Proof of concept, this can be HUGELY sped up (too many allocations)
//! By convention we start at state 0,
//! also all words are 5 char longs so we don't need
//! to track ending states and what not.
//! 
//! 
//! 
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::fs;
use std::io::prelude::*;
use std::io::BufReader;
use rayon::prelude::*;
use std::time::Instant;

#[derive(PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord, Debug, Default)]
/// Strongly typed state because *I will messup stuff* in the code and this
/// catches some errors and make things more explicit
struct State(pub usize);

#[derive(Default)]
/// Also we will always work with DETERMINISTIC FSA, as
/// the hasmap key implies.
struct DSA {
    transitions: HashMap<(State, char), State>,
    state_id: State,
}

#[derive(Default)]
/// Maybe non deterministic FSA.
struct FSA {
    transitions: HashMap<State, HashMap<char, Vec<State>>>,
    /// this always point to the next state id to add
    state_id: State,
}

impl DSA {

    pub fn to_dot(&self) -> String {
        let mut result = "digraph {\n".to_string();

        for ((src, c), dst) in self.transitions.iter() {
            result.push_str(format!(
                "\t{} -> {} [label=\"{}\"]\n", 
                src.0, dst.0, c
            ).as_str());
        }

        result.push_str("}\n");
        result
    }

    /// Convert the DSA to an FSA so that we can merge it with another
    pub fn to_fsa(self) -> FSA {
        let mut transitions: HashMap<State, HashMap<char, Vec<State>>> = HashMap::with_capacity(self.transitions.len());

        for ((src, c), dst) in self.transitions {
            transitions.entry(src).or_default()
                .entry(c).or_default()
                .push(dst);
        }

        FSA { 
            transitions,
            state_id: self.state_id, 
        }
    }

    /// This can greatly optimized changing datastructures but for a POC is
    /// enought
    fn get_transitions_from_state(&self, state: State) -> Vec<(char, State)> {
        self.transitions.iter()
            .filter(|((src, _), _)| *src == state)
            .map(|((_, c), dst)| (*c, *dst))
            .collect()
    }

    /// Minimize the FSA, we don't **really** care about dead states elimination
    /// because we can only insert strings so we cannot generate dead states.
    /// Therefore we will just proceede to merge equivalent states using a 
    /// naive implementation of a Moore's like algorithm
    pub fn minimize(&mut self) {
        let mut eq_matrix = EquivalenceMatrix::build(&self);
        let eq_classes = eq_matrix.get_equivalence_classes();

        // traduce the old DSA states to the new minimized ones, this will 
        // insert multiple times the duplicated states, but the result will be
        // the same so we don't care for correctness sake
        let mut new_transitions = HashMap::with_capacity(self.transitions.len());
        for ((src, c), dst) in self.transitions.iter() {
            let src = eq_classes.get(src).unwrap_or(src);
            let dst = eq_classes.get(dst).unwrap_or(dst);
            new_transitions.insert((*src, *c), *dst);
        }

        // set the new traduced states
        self.transitions = new_transitions;

        // rename the states so that they are contiguous
        //self.densify();
    }

    /// densify the states so that they are in a dense range
    fn densify(&mut self) {
        let mut state_id = 0;
        let mut map = HashMap::with_capacity(self.transitions.len());
        let mut cleaned_transitions = HashMap::with_capacity(self.transitions.len());

        for ((src_state, c), dst_state) in self.transitions.iter() {
            // remap the src state
            let src_id = match map.entry(src_state) {
                Entry::Occupied(src_id) => {
                    *src_id.get()
                },
                Entry::Vacant(src_id) => {
                    let new_state = State(state_id);
                    src_id.insert(new_state);
                    state_id += 1;
                    new_state
                }
            };
            // remap the dst state
            let dst_id = match map.entry(dst_state) {
                Entry::Occupied(dst_id) => {
                    *dst_id.get()
                },
                Entry::Vacant(dst_id) => {
                    let new_state = State(state_id);
                    dst_id.insert(new_state);
                    state_id += 1;
                    new_state
                }
            };

            // add the densified rule
            cleaned_transitions.insert((src_id, *c), dst_id);
        } 

        self.transitions = cleaned_transitions;
        self.state_id = State(state_id);
    }
}

impl FSA {
    /// add a word to the fsa
    pub fn add_word(&mut self, word: &str) {
        let mut state_to_explore = State(0);
        for c in word.chars() {
            // allocate a new state
            self.state_id.0 += 1;
            // add the transition to the new node
            self.transitions.entry(state_to_explore).or_default()
                .entry(c).or_default()
                .push(self.state_id);
            // add the new char to the state we just added
            state_to_explore = self.state_id;
        }
    }

    /// merge two FSA in the trivial manner, rename all the nodes of the other
    /// FSA and merge the starting states
    pub fn merge(mut self, other: FSA) -> FSA {
        for (mut src, destinations) in other.transitions {
            // create a new node_id for all nodes except the 0 which will be
            // common for bot fsa
            if src != State(0) {
                src.0 += self.state_id.0;
            }
            // get and insert the new state transitions
            let dsts = self.transitions.entry(src).or_default();
            // add all the transitions
            for (c, dst_states) in destinations {
                // create or add the transitions to the nodes
                dsts.entry(c).or_default().extend(
                    // translating the dst nodes to the new ids
                    dst_states.iter().map(|s| 
                        if *s != State(0) {
                            State(s.0 + self.state_id.0)
                        } else {
                            *s
                        })
                );
            }
            
        }

        // add the number of states, this will be the sum of the two -1 because
        // we will keep the same starting state
        self.state_id.0 += other.state_id.0.saturating_sub(1);

        self
    }   

    /// Convert the FSA to a DSA, this also does dead states elimination as a
    /// side-effect. We are going to explore in parallel all the states, this
    /// can result in states explosion as highly non deterministic FSA with
    /// n states can result in DSA with 2**n states.
    pub fn determinize(self) -> DSA {
        // the new DSA data
        let mut state_id = 0;
        let mut dsa_transitions = HashMap::with_capacity(self.transitions.len());
        
        // we start from the 0 state and explore in parallel each state, keeping
        // track of the combination of states we explore
        let mut states_to_explore = Vec::new();
        states_to_explore.push(vec![State(0)]);

        // translation between states sets and state_ids in the DSA, also this
        // can be used for deduplication
        let mut map = HashMap::with_capacity(self.transitions.len());

        while let Some(states) = states_to_explore.pop() {
            // get a node_id for the current set state
            let src_id = *map.entry(states.clone())
                .or_insert_with(|| {
                    let result = state_id;
                    state_id += 1;
                    State(result)
                });
            
            // TODO!: if the substate is just 1 node we can skip this merging
            // Compute the **merged** transitions of all the states
            let mut merged_state_transitions: HashMap<char, Vec<State>> = HashMap::with_capacity(self.transitions.len());
            for sub_state in states.iter() {
                // if the substate has transitions (eg is not a final state)
                if let Some(node_transitions) = self.transitions.get(sub_state) {
                    // add its transitions to the merged one
                    for (c, dsts) in node_transitions {
                        merged_state_transitions.entry(*c).or_default()
                            .extend(dsts);
                    }
                }
            }
            // create the DSA transitions from the now merged state
            for (c, mut dsts) in merged_state_transitions {
                // sort the dsts because otherwise we could have duplicates if
                // the states are in different order
                dsts.sort();
                // create a new id for the dsts (if not already present)
                let dsts_id = *map.entry(dsts.clone())
                    .or_insert_with(|| {
                        let result = state_id;
                        state_id += 1;
                        states_to_explore.push(dsts);
                        State(result)
                    });

                // add the transition
                dsa_transitions.insert(
                    (src_id, c), dsts_id
                );
            }
        }
        
        DSA { 
            transitions: dsa_transitions, 
            state_id: State(state_id) 
        }
    }

    pub fn to_dot(&self) -> String {
        let mut result = "digraph {\n".to_string();

        for (src, transitions) in self.transitions.iter() {
            for (c, dsts) in transitions.iter() {
                for dst in dsts {
                    result.push_str(
                        format!(
                            "\t{} -> {} [label=\"{}\"]\n", 
                            src.0, dst.0, c
                        ).as_str()
                    );
                }
            }
        }

        result.push_str("}\n");
        result
    }
}

/// A sparse matrix of the possible equivalent states for each state,
/// this could be dense but I'm lazy lol
#[derive(Debug)]
struct EquivalenceMatrix(HashMap<(State, State), Vec<(State, State)>>);

impl EquivalenceMatrix {
    pub fn build(dsa: &DSA) -> Self {
        let max_state = dsa.state_id.0;
        let mut eq_matrix = EquivalenceMatrix(HashMap::with_capacity(dsa.transitions.len()));

        // compare each pair of states and check if they could be equivalent or
        // they differ for sure.
        // The hasmap will have the following semantic: the keys will always be
        // with (lower, bigger) of the two states (since it's simmetric).
        // If a key is not present, then the two states are not compatible.
        // If a key has as value an empty vector, then the two state are 
        // equivalent.
        // If a key has as value a non empty vector, then the two state are 
        // equivalents iff also all the tuples in the vec are equivalent.
        // So to get the equivalence classes we will propagate the equivalences 
        // until convergence, at this point we will either have states with 
        // empty vectors or cycles of dependancies, which are also equivalences.
        for src_id in 0..max_state {
            let mut src_transitions = dsa.get_transitions_from_state(State(src_id));
            src_transitions.sort();

            // the equivalence matrix is simmetric so we can just compute the 
            // upper triangular part of it
            'outer: for dst_id in src_id + 1..max_state {
                let mut dst_transitions = dsa.get_transitions_from_state(State(dst_id));

                // since it's a DFA, if the number of transitions don't match
                // it means that there have to be at least a transition that
                // differs them
                if src_transitions.len() != dst_transitions.len() {
                    // continue so that we don't add the pair to the hashmap
                    // and we speed up the propagation later
                    continue 'outer;
                }
                dst_transitions.sort();

                let mut dependencies = Vec::new();
                'inner: for ((c1, s1), (c2, s2)) in src_transitions.iter().zip(dst_transitions.iter()) {
                    // the transitions are sorted by char, so if they differ
                    // it means that at least 2 transitions (one for each state)
                    // are different, so these two nodes cannot be equivalent 
                    if c1 != c2 {
                        continue 'outer;
                    } 

                    // if the transition is the same then everything is fine
                    // and it's not a dependancy
                    if s1 == s2 {
                        continue 'inner;
                    }

                    // since we want to save the upper triangular matrix, we can
                    // re-order the states so that the lookups will always hit
                    // if present
                    let smaller = *s1.min(s2);
                    let bigger = *s1.max(s2);

                    // save the dependancy
                    dependencies.push((smaller, bigger));
                }

                // now that we know that the state might be equivaent we add this
                // to the matrix
                eq_matrix.0.insert((State(src_id), State(dst_id)), dependencies);
            }
        }

        // the reamining state with non empty vecs are equivalent.
        eq_matrix
    }

    /// do an iteration of propagation and return if anything was modified
    fn propagate_iter(&mut self) -> bool {
        // maybe we can do this in place but currently idk how :shrugs:
        // allocations are bad mhkay?
        let mut changed_something = false;
        let mut new_eq_matrix = HashMap::with_capacity(self.0.len());

        // for each possible equivalence, check the deps and propagate constants
        'outer: for ((s1, s2), deps) in self.0.iter() {
            let mut new_deps = Vec::new();

            'inner: for dep in deps {
                // check if the dep is possible, true, or false
                match self.0.get(dep) {
                    Some(sub_dep) => {
                        // the depndancy is satisifed so we won't add it to the
                        // new matrix
                        if sub_dep.is_empty() {
                            changed_something = true;
                            continue 'inner;
                        } 

                        // we cannot say anything new about the dep so we just
                        // add it
                        new_deps.push(dep.clone());
                    }
                    // if the dependancy is not a possible eq, then also this
                    // is not a possible eq, so we "remove" the cell by not
                    // inserting it in the new transitions
                    None => {
                        changed_something = true;
                        continue 'outer;
                    }
                }
            }

            // add the new dependancies
            new_eq_matrix.insert((*s1, *s2), new_deps);
        }

        self.0 = new_eq_matrix;
        changed_something
    }

    // propagate until convergence
    fn propagate(&mut self) {
        while self.propagate_iter() {}
    }

    pub fn get_equivalence_classes(&mut self) -> HashMap<State, State> {
        // ensure that we are at convergence
        self.propagate();

        let mut set_id = 0;
        let mut eq_classes = HashMap::with_capacity(self.0.len());
        let mut eq_classes_rev_map = HashMap::with_capacity(self.0.len());

        for (s1, s2) in self.0.keys() {
            let res_s1 = eq_classes_rev_map.get(s1).cloned();
            let res_s2 = eq_classes_rev_map.get(s2).cloned();
            match (res_s1, res_s2) {
                // both don't exist, we just create a new set
                (None, None) => {
                    eq_classes.insert(set_id, vec![*s1, *s2]);
                    eq_classes_rev_map.insert(*s1, set_id);
                    eq_classes_rev_map.insert(*s2, set_id);
                    set_id += 1;
                }
                // if only one exist, add it to the already present set
                (Some(set), None) => {
                    eq_classes_rev_map.insert(*s2, set);
                    eq_classes.get_mut(&set).unwrap().push(*s2);
                }
                // if only one exist, add it to the already present set
                (None, Some(set)) => {
                    eq_classes_rev_map.insert(*s1, set);
                    eq_classes.get_mut(&set).unwrap().push(*s1);
                }
                // Both exists, now we have to merge the sets and this sucks
                (Some(set_1), Some(set_2)) => {
                    if set_1 == set_2 {
                        continue;
                    }
                    let deps = eq_classes.get(&set_2).unwrap().clone();
                    eq_classes.get_mut(&set_1).unwrap().extend(deps.iter());
                    eq_classes.remove(&set_2);
                    for dep in deps {
                        eq_classes_rev_map.insert(dep, set_1);
                    }
                },
            }
        }

        // create a remapping state map to the minimum
        let mut result = HashMap::with_capacity(eq_classes.len());
        for eq_class in eq_classes.values() {
            let id = eq_class.iter().min().unwrap();
            for state in eq_class {
                result.insert(*state, *id);
            }
        }

        result
    }
}

fn main() {
    let start = Instant::now();
    let fd = BufReader::new(fs::File::open("./wordlist.txt").unwrap());
    let mut res_file = fs::File::create("./result.dot").unwrap();

    let words = fd.lines().map(Result::unwrap).collect::<Vec<_>>();
    
    let dsa = words.par_chunks(64).map(|slice| {
        let mut fsa = FSA::default();
        for word in slice {
            fsa.add_word(word.as_str());
        }
        let mut dsa = fsa.determinize();
        dsa.minimize();
        dsa
    }).reduce( DSA::default,
        |dsa1, dsa2| {
        let fsa1 = dsa1.to_fsa();
        let fsa2 = dsa2.to_fsa();
        let merged = fsa1.merge(fsa2);
        let mut dsa = merged.determinize();
        dsa.minimize();
        dsa
    });

    println!("it took: {:.4}s", start.elapsed().as_secs_f64());

    let dot = dsa.to_dot();
    println!("{}", dot);

    println!("{} edges and {} nodes", dsa.transitions.len(), dsa.state_id.0);

    res_file.write_all(dot.as_bytes()).unwrap();

}