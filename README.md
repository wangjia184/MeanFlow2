## Derivation of the MeanFlow Identity

### Step 1: Definition of Average Velocity
The average velocity $u(z_t, r, t)$ over the time interval $[r, t]$ is defined as:
```math
u(z_t, r, t) \triangleq \frac{1}{t-r} \int_r^t v(z_\tau, \tau) d\tau
```
where:
- $z_t$ is the state at time $t$
- $v(z_\tau, \tau)$ is the instantaneous velocity field
- $t > r$ (time interval is positive)

```
  0    ≤    r   <    t    ≤   1
  |---------------------------|
  |<-- r -->|
  |<------ t ------->|
```

---

### Step 2: Rearranged Definition
Multiply both sides by $(t-r)$:
```math
(t-r) u(z_t, r, t) = \int_r^t v(z_\tau, \tau) d\tau
```



---

### Step 3: Total Derivative with Respect to $t$
Take the total derivative of both sides with respect to $t$ (note $z_t$ depends on $t$):

**Left Side** (Product Rule):
```math
\frac{d}{dt} \left[(t-r) u(z_t, r, t)\right] = u(z_t, r, t) + (t-r) \frac{d}{dt} u(z_t, r, t)
```

**Right Side** (Fundamental Theorem of Calculus):
```math
\frac{d}{dt} \int_r^t v(z_\tau, \tau) d\tau = v(z_t, t)
```



---

### Step 4: Expand Total Derivative of $u$
The total derivative of $u(z_t, r, t)$ is:
```math
\frac{d}{dt} u(z_t, r, t) = \underbrace{\frac{\partial u}{\partial z_t} \cdot \frac{dz_t}{dt}}_{\text{Chain rule}} + \frac{\partial u}{\partial t}
```

Since $\frac{dz_t}{dt} = v(z_t, t)$ (by ODE definition) and $r$ is constant ($\frac{dr}{dt} = 0$):
```math
\frac{d}{dt} u(z_t, r, t) = \frac{\partial u}{\partial z_t} v(z_t, t) + \frac{\partial u}{\partial t}
```



---

### Step 5: Establish Equality
Combine results from Steps 3 and 4:
```math
u(z_t, r, t) + (t-r) \left( \frac{\partial u}{\partial z_t} v(z_t, t) + \frac{\partial u}{\partial t} \right) = v(z_t, t)
```


---

### Step 6: MeanFlow Identity
Rearrange to obtain the final identity:
```math
\boxed{u(z_t, r, t) = v(z_t, t) - (t-r) \left( \frac{\partial u}{\partial z_t} v(z_t, t) + \frac{\partial u}{\partial t} \right)}
```



---

## Key Mathematical Insights
1. **Total Derivative Necessity**  
   The chain rule accounts for implicit ($z_t$) and explicit ($t$) dependencies:
   ```math
   \frac{du}{dt} = \nabla_{z_t} u \cdot \frac{dz_t}{dt} + \frac{\partial u}{\partial t}
   ```

2. **Boundary Condition**  
   When $r \to t$:
   ```math
   \lim_{r \to t} u(z_t, r, t) = v(z_t, t)
   ```
   This ensures consistency with the instantaneous velocity field.

3. **Time Symmetry**  
   The derivation holds for $r > t$ by using $|t-r|$ in the denominator.