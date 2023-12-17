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
          inputsFrom = [ self.packages.${system}.src ];
          # Still want h5py available. For now, python311packages.h5py has the
          # version 3.10.0 which is the same specified in poetry.lock.
          packages = with pkgs; [ poetry python311Packages.h5py ];
          # Linking shared libraries for zlib is broken, this points nix to the
          # correct place.
          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
          '';
        };
      });
}
