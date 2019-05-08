# compile rust libs
cd rs
for repo in ./*; do
  cd $repo
  name=$(basename $repo)
  cargo build --release
  cp target/release/lib$name.so ../../$name.so
  cd ..
done
cd ..

# put the compiled libs where they should end up
mv test_rs.so bench/

cp nc1nn.so tests/nc1nn/
mv nc1nn.so bench/nn/
