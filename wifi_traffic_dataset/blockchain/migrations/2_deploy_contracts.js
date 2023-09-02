const SimpleStorage = artifacts.require("ModelRegistry");

module.exports = function(deployer) {
  deployer.deploy(SimpleStorage);
};
