% Multi Core Parallel implementation of 
% Simplified Sakura.
% Authors Prasad Rao, Stuart Haber, HP Labs
% Written in the Literate Haskell Mode

\begin{frame}{Highlights}
\begin{itemize}
\item  Type definition for shape: Slide \ref{slide-shape}
\item  Serial Tree Hash:Slide \ref{slide-serial-hash}
\item  Parallel Tree Hash: Slide \ref{slide-parallel-hash}
\item  Tree Hash Modes: Slide \ref{slide-modes}
\end{itemize}
\end{frame}

\begin{frame} [fragile]{Preliminaries}

\begin{itemize}

\item Import needed classes

\begin{code}
import Data.Word8
import qualified Codec.Binary.UTF8.String as CodStr
\end{code}

\end{itemize}
\begin{itemize}

\item
For parallelism add \ldots
\begin{code}
import Control.Parallel.Strategies
import Data.Time.Clock
\end{code}
\end{itemize}

\begin{itemize}

\item Crypto Imports
\begin{code}
--import Crypto.Hash.SHA256
--import Data.Digest.SHA256
import Data.Digest.SHA384
import System.Environment
\end{code}

\item Type abbreviations
\begin{code}
type BStr = [Word8]
type HashF = [Word8] -> [Word8]
\end{code}


\end{itemize}

\end{frame}
%--------------------------------------------------------

%\subsection{Core Stuff}

\begin{frame} [fragile,label=slide-shape]{Type Definition: Shape}
\begin{code}
data HShape =
  InnerHash HShape
  |Concat [HShape]
  |Interleaving [HShape]
  |Slice Int Int
  |Pad  BStr
  deriving Show
\end{code}
\pause

\underline{Complete} w.r.t. the Sakura specification.  

\pause

Meaning of the type: Entities on the right hand side are functions.

\begin{verbatim}
InnerHash : HShape -> HShape
Concat: [HShape] -> HShape
Interleaving: [HShape] -> HShape
Slice: Int -> Int -> HShape
Pad: ByteString -> HShape
\end{verbatim}
\end{frame}
%\subsection{The Important Stuff}


%--------------------------------------------------------
\begin{frame}[fragile,label=slide-serial-hash]{Serial Tree Hash}
\begin{code}
s :: HashF -> HShape -> BStr -> BStr
\end{code}
\pause
Only place where the hash function is called
\begin{code}
s h (InnerHash aShape) bStr  =  h (s h aShape bStr)
\end{code}
\pause
Concatenate results of subtree computations \uncover<5->{ -- \alert{***}}
\begin{code}
s h (Concat l) bStr = concat  (map (\x-> s h x bStr)  l)
\end{code}
\pause
Only way to directly consume input string
\begin{code}
s _ (Slice from to) bStr = my_slice from to bStr
\end{code}
\pause
Only way to directly insert padding bits
\begin{code}
s _ (Pad x) _ = x
\end{code}
\end{frame}
%-----------------------------------------------------------
\begin{frame}[fragile,label=slide-parallel-hash]{Parallel Tree Hash}
\begin{code}
p :: HashF -> HShape -> BStr -> BStr
\end{code}
{{\color{blue}} Only place where the hash function is called}
\begin{code}
p h (InnerHash aShape) bStr = h $ p h aShape  bStr
\end{code}
Concatenate results of subtree computations -- \alert{parallel code}
\begin{code}
p h (Concat l) bStr = concat (parMap rpar (\mu -> p h mu bStr) l)
\end{code}
Only way to directly consume input string
\begin{code}
p _ (Slice from to) bStr = my_slice from to bStr
\end{code}
\pause
Only way to directly insert padding bits
\begin{code}

p _ (Pad x) _ = x
\end{code}
\end{frame}
%--------------------------------------------------------
%\subsection{Modes}
%\subsubsection{Fixed Block Length}
\begin{frame}[fragile,label=slide-modes]{Modes}
Mode: Mapping from size to shape.
\end{frame}
\begin{frame}[fragile]{Modes: Fixed Block Length Mode}
\begin{code}
chunker :: Int -> Int -> BStr -> BStr -> HShape
chunker n size innerpad rootpad =
  let b = quot n size
      make_node i = InnerHash (fb_pad (fb_ith i size) innerpad)
      ranges = map make_node [0 .. b]
      all_ranges = if rem n size == 0 then
                   ranges
                 else
                    ranges ++ [InnerHash 
                               (fb_pad (fb_last b size n) innerpad)]
  in
   InnerHash (Concat (all_ranges ++ [(Pad rootpad)]))
\end{code}
\end{frame}

\begin{frame}[fragile]{Binary Tree Hash}
\begin{code}
c2w8 :: String -> [Word8]
c2w8 = CodStr.encode
\end{code}



\begin{frame}[fragile]{Modes: Fixed Block Length Mode}
\begin{code}
a_block_mode x block_size  =
    chunker (length x) block_size  (c2w8 "IIII") (c2w8 "RRRR")


\end{code}
\end{frame}

%-------------------------------------------------------------------------------
%\subsection{Helpers}
\begin{frame}[fragile]{Slicer}
A wrapper around a subsequence operation of ByteStrings. 

\begin{code}

my_slice :: Int -> Int -> BStr -> BStr
my_slice from to = (drop from).(take to)

\end{code}
\end{frame}

\begin{frame}[fragile]{helpers}
\begin{code}

fb_ith i size = Slice (i * size) ((i + 1) * size)

fb_last i size n = Slice (i*size) n

fb_pad chunk innerpad = Concat [chunk, (Pad innerpad)]

\end{code}
\end{frame}


%\subsubsection{Binary Trees}
\begin{frame} [fragile]{Shape a list into a binary tree}
\begin{code}
data BTree a = Inner (BTree a) (BTree a)
             |Leaf a
  deriving Show
\end{code}
\pause

\begin{code}
one_level_treefy :: [BTree a] -> [BTree a]
one_level_treefy (h0:h1:tl) = (Inner h0 h1):(one_level_treefy tl)
one_level_treefy x = x
\end{code}
\pause

\begin{code}
fixpt_treefy  :: [BTree a] -> BTree a
fixpt_treefy [x] = x
fixpt_treefy  x_l = fixpt_treefy ( one_level_treefy x_l)

to_tree :: [a] -> BTree a
to_tree = fixpt_treefy . (map Leaf)
\end{code}

\end{frame}

\begin{frame}[fragile]{Apply Padding and Inner Hashing to a vanilla tree to turn it into a shape}
\begin{code}
pad_and_hash (Leaf slice) leaf_pad inner_pad root_pad =  InnerHash (Concat [slice, Pad leaf_pad])
pad_and_hash (Inner t1 t2) leaf_pad inner_pad  root_pad = InnerHash (Concat [
             pad_and_hash t1 leaf_pad inner_pad inner_pad,
             pad_and_hash t2 leaf_pad inner_pad inner_pad,
             Pad root_pad])
\end{code}

\end{frame}


\begin{frame}
We are now ready to compute the binary tree hash {\tt bin\_tree\_hash} of a {\tt ByteString}.
\begin{code}
bin_tree_hash  :: (BStr -> BStr) -> Int -> BStr -> BStr
bin_tree_hash inner_hash block_size a_str =
              let 
              len = length a_str
              ranges = size_to_pairs len block_size
              shape = to_shape ranges (c2w8 "LLLL")  (c2w8 "III ") (c2w8  "RRRR")
              in
              s inner_hash shape a_str

p_bin_tree_hash  :: (BStr -> BStr) -> Int -> BStr -> BStr
p_bin_tree_hash inner_hash block_size a_str = 
              let 
                  len = length a_str
                  ranges = size_to_pairs len block_size
                  shape = to_shape ranges (c2w8 "LLLL")  (c2w8 "III ") (c2w8  "RRRR")
              in
                p inner_hash shape a_str
\end{code}
\end{frame}

\begin{frame}[fragile]{Generate leaves}
Convert a number representing a size into a sequence of intervals that represent equally sized blocks.
\pause
\begin{code}
size_to_pairs :: Integral a => a -> a -> [(a, a)]
size_to_pairs n block_size = let 
              b = quot n block_size
              ranges =  map (\a -> (a * block_size, (a + 1) * block_size)) [0 .. b]
              in
              if rem n block_size == 0 then
              ranges
              else
              ranges ++ [(b * block_size, n)]
\end{code}
\end{frame}

\begin{frame}[fragile]{Mapping over a tree II}
\begin{code}


to_shape :: [(Int,Int)] -> BStr -> BStr -> BStr -> HShape
to_shape ranges leaf_pad inner_pad root_pad = let
         leaves =  map (\(x,y) -> (Slice x y)) ranges
         slice_tree = to_tree leaves
         in
         pad_and_hash slice_tree leaf_pad inner_pad root_pad
\end{code}
\end{frame}


\begin{frame}[fragile]{Exerciser}

\begin{code}
my_id x = x
ipad = Pad (c2w8 "ipad")
rpad = Pad (c2w8 "rpad")
aShape = (Concat [(InnerHash (Slice 0 4)), (InnerHash (Slice 4 8))])
b1 = my_slice 0 4 (c2w8 "abcd")
b0 = (Slice 0 4)
b0e = s my_id b0 (c2w8 "abcdefgh")
shape2 = (Concat [Concat [(InnerHash (Slice 0 4)),ipad] , Concat [(InnerHash (Slice 4 8)),ipad],rpad])
aHash = s my_id aShape (c2w8  "abcdefgh")
hshape2 = s my_id shape2 (c2w8 "abcdefgh")


alongstring = "abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789"

balongstr = c2w8 alongstring
lshape = chunker (length balongstr) 7 (c2w8 "IIII") (c2w8 "RRRR")
balonghash = s my_id lshape balongstr
treeHash = bin_tree_hash my_id 10 (c2w8 alongstring)

parHash1 aShape =  p my_id aShape balongstr
  


test1 = parHash1 lshape ==  balonghash
test2 = p_bin_tree_hash my_id 10  (c2w8 alongstring) == treeHash

\end{code}
\end{frame}

%\subsection[Main]{Main}
\begin{frame}[fragile]{Main Stub}
 
\begin{frame}[fragile]{Main Stub for Parallel Hash}
\begin{code}
do_hash func = do
    args <- getArgs
    putStr "Args:"
    putStrLn (show args)
    contents <- readFile (head args)
    putStrLn $ "Hash:" ++ (show (func hash (a_block_mode contents (read (args !! 1)::Int)) (c2w8 contents)))

main = do_hash p
-- c for serial p for parallel
\end{code}
\end{frame}