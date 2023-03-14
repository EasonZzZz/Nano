class DNAReference:
    """
    A class to hold a reference genome and provide some basic
    functionality for working with it.
    """
    def __init__(self, ref_path):
        self.chrom_names = []
        self.chrom_lengths = {}
        self.chrom_seqs = {}
        with open(ref_path) as ref_file:
            for line in ref_file:
                if line.startswith(">"):
                    chrom_name = line.strip()[1:]
                    self.chrom_names.append(chrom_name)
                    self.chrom_lengths[chrom_name] = 0
                    self.chrom_seqs[chrom_name] = ""
                else:
                    self.chrom_seqs[chrom_name] += line.strip()
                    self.chrom_lengths[chrom_name] += len(line.strip())

    def get_chrom_length(self, chrom_name):
        """
        Get the length of a chromosome.
        :param chrom_name: The name of the chromosome.
        :return: The length of the chromosome.
        """
        return self.chrom_lengths[chrom_name]

    def get_chrom_names(self):
        """
        Get the names of the chromosomes.
        :return: A list of the chromosome names.
        """
        return self.chrom_names

    def get_chrom_seq(self, chrom_name):
        """
        Get the sequence of a chromosome.
        :param chrom_name: The name of the chromosome.
        :return: The sequence of the chromosome.
        """
        return self.chrom_seqs[chrom_name]

