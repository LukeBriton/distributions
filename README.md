# *Vector-Valued Distributions and Fourier Multipliers*, Herbert Amann

## What's this?

My journey trying to get a perfect PDF version of Amann's distributions.pdf without its original .tex file. The discovery of its .ps "IR" code is coincidential, but it turns out to be the best way to make it through.

Also, hope I could persisit in reading this book some day... Prof. Amann wrote splendid Analyses I, II, III, which I still haven't finished yet. 

## The book

### Pre-preface (?)
> This manuscript was originally planned to be Chapter VI, that is, the first chapter of Volume II of the treatise whose first volume is [Ama95]\(Note: *Linear and Quasilinear Parabolic Problems, Volume I: Abstract Linear Theory*). In the meantime I have changed the concept of that volume and do not plan to include this material. For this reason, and since some of the results presented here have already been cited by several authors being in possession of preliminary versions, I have decided to make it publicly available by putting it on my home page. Zürich, July 2003

### Preface
> In the study of evolution equations by functional-analytical techniques one is led quite naturally to the study of pseudodifferential operators with operator-valued symbols. If one wants to avoid being restricted to a Hilbert space setting — that would be much too narrow for nonlinear problems — one has to be able to handle efficiently distributions with values in general Banach spaces. In particular, one has to be in possession of a good theory of convolutions for the case that both factors are Banach-space-valued distributions. For this reason, specializing the much more general results of L. Schwartz to a Banach space setting, we develop in the first section the theory of vector-valued distributions. Since there seems to be no other complete exposition of this theory easily available we try to be rather complete, thus going beyond the minimum we really need.
> 
> Vector-valued distributions are most useful in connection with techniques of harmonic analysis in a Banach space setting. In particular, we prove some Fourier multiplier theorems involving operator-valued symbols. These results, combined with dyadic decompositions in the Fourier image, are of fundamental importance. They are developed in Section 2. It is an important fact that, unlike the Banach space version of the Mikhlin multiplier theorem, our multiplier theorems do not require any restriction on the class of Banach spaces, like the UMD property, for example.
> 
> In the last section we first give a simple application of vector-valued distribution theory to linear evolution equations. Namely, we introduce the concept of distributional solutions and give sufficient conditions for their existence and uniqueness. The rest of that section is then essentially devoted to a detailed study of the Gauß-Weierstraß semigroup and related subjects. Its results are fundamental for an in-depth study of function spaces.

## Directory Tree

- **backup**: containing original files.
    - [**distributions.pdf**](https://user.math.uzh.ch/amann/files/distributions.pdf): which is uncopyable, with garbled text when copy pasting, bitmapped rather than vector.
    - [**distributions.ps**](https://user.math.uzh.ch/amann/files/distributions.ps): PostScript file, somehow an intermediate representation before compiling to PDF, containing quite enough information to reconstruct a better pdf, though not without effort.    
    Source: [reference request - English translation of Schwartz's papers on vector-valued distributions - MathOverflow](https://mathoverflow.net/questions/386348/english-translation-of-schwartzs-papers-on-vector-valued-distributions)

- **OCR-ABBYY**: containing original, redacted (text layer removed), and OCRed (English; Simple math formulas) PDFs.

    The OCRed one is copyable, mostly good for text, but not good enough for formulae.

- **OCR-Nougat**: trying to reconstruct the book with better specialized OCR into markdown (then maybe to pdf).

    Found this method from [Ultimate solution for OCR of Maths Textbooks? : r/MLQuestions](https://www.reddit.com/r/MLQuestions/comments/1hlj2vc/ultimate_solution_for_ocr_of_maths_textbooks/), after encountering several problems configuring the github version, got inspired by [TypeError: BARTDecoder.prepare_inputs_for_inference() got an unexpected keyword argument 'cache_position' · Issue #213 · facebookresearch/nougat](https://github.com/facebookresearch/nougat/issues/213#issuecomment-2021190588) to turn to Transformers one.
    Note that Nougat is quite efficient, taking less than 2GiB VRAM (2134MiB for nougat-base), and about 36 min for the 148 pages on a single RTX 3090. Even though the output needs much proofreading (strange repetitions, ignorance of commutative diagrams), it would still lessen one's burden to type in numerous $\LaTeX$ formulae.

- **PostScript-PSviewer**: converting the postscript file to a PDF file by PSviewer of TeX Live, comprehensible in a way, still containing a lot of nonsense when copied, still bitmapped.

- **PostScript-pkfix**: with the help of pkfix-helper, pkfix, and ps2pdf, to finally generate a copyable perfect (in fonts, formulae, diagrams, ligatures, latin extended additional etc.) PDF with vector fonts. 
    - **raw**: no option offered, bad in mathematical symbols(font style, diagram arrow, positions), cannot print fi, ff, ffi, fl (etc.) correctly, dropping á, é, ä, ö, ü and ß etc.
    - **half-cooked**: forcing those whose fonts' vector versions are existent according to kpsewhich, now better at mathematical symbols.
    - **cooked**: utilzing cm-super-t1.map to force those with aliases to T1-encoded CM-Super fonts.    
    PS: ChatGPT-5.2 Thinking almost fooled me by dropping some from the list accidentally (or purposely?) that some's fonts were worse than half-cooked.

- **distributions_fixed.pdf**: symlink to the cooked pdf, final version for the moment.
- **distributions_fixed.ps**: symlink to the cooked ps, final version for the moment.