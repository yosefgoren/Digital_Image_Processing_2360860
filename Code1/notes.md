## Part a:


## Part b:
### 6
Placing the code for this in `b.py`.
I can run the animation with `python3 b.py`.
Looks like spongbob running in circles:

![alt text](image.png)

### 7
This was more tricky than I expected.
First, diff the first frame from all frames and normalize the result: this makes it easy to tell which frame is identical to the first frame (when we see all back).
If I run `python3 b.py 200 True` I see diff frames slowly advancing:

![alt text](image-1.png)

Now I can see the first replay of this first frame is around frame #20 but it's still a hastle timing it exactly.
So I add a feature for moving between the frames manually and print the next frame number each time.
If I run with `python3 b.py 0 True --manual` I see:

![alt text](image-2.png)

Advancing to frame at index 19 shows:

![alt text](image-3.png)

So indices trictly included in a single cycle are `[0, 18]` including edges - which is 19 different indices.

So there are $19 [Frames / Cycle]$, if every frame was displayed for one second - spongebob would be cycling at $1/19 [Hz]$.

### 8
We would have to sample at least $2/19 [Samples / Frame]$.
More explicitly, if we set:
* $F = \frac{1}{19} [Cycle / Frame]$
* $N = n [Frame / Second]$

We can find:
$$
    f_{max}=F*N = \frac{n}{19} \left[\frac{Cycle}{Frame} \cdot \frac{Frame}{Second}\right] = \frac{n}{19} [Hz]
$$
$$
\Rightarrow f_{nyquist} = 2*f_{max} = \frac{2n}{19} [Hz]
$$
