{
  description = "Deep Learning Project";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication defaultPoetryOverrides;
      in
      {
        packages = {
          src = mkPoetryApplication { 
            projectDir = self;
            # h5py in poetry2nix overrides is broken, null ensures that the poetry2nix
            # overrides are not applied
            overrides = defaultPoetryOverrides.extend (self: super: { h5py = null; });
          };
          default = self.packages.${system}.src;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [ 
            poetry
            
            python311
            pre-commit
            python311Packages.black
            python311Packages.isort
            
            python311Packages.scipy
            python311Packages.scikit-learn
            python311Packages.h5py
            python311Packages.matplotlib
            python311Packages.tensorflow
            python311Packages.keras
            python311Packages.pytorch
            python311Packages.pandas
            keras-tcn
          ];
          
          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
          '';
        };
      });
}
